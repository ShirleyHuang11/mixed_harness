"""Seed harnesses for the tau2-bench retail domain: the routed harness and its two single-path controls.

Routed graph (one turn), in dependency order:

    compile (python)   parse the conversation, keep fetched entities, build the legal-action catalog A(s), fill what
                       code can fill, list what it cannot (`needs`), flag the step closed or open, name its cluster,
                       and write every question Jev is asked this turn
    judge (jev)        one request: route pick (closed / narrow / escalate), action pick, a reordered paraphrase of
                       the action pick (agreement signal), "does any option fit", "did the user confirm", and the
                       candidate questions (which item, which variant, which payment method, which name)
    route (python)     the routing layer: compiler flag first (open -> narrow), then Jev's pick, then the confidence
                       threshold tau on the action pick (below it -> escalate), then agreement fall-through (-> narrow),
                       then the per-cluster policy and its epsilon shadow share, then trajectory-level escalation
    decision (python)  the closed path: Jev's answers -> a tool call with code-filled arguments, or a template reply
    menu (llm)         the narrow path, step 1: the LLM writes 2-4 candidate actions (skipped unless routed narrow)
    menu_q (python)    turns the menu into a choice question (null when there is no menu)
    menu_pick (jev)    the narrow path, step 2: Jev picks from the menu (skipped when menu_q is null)
    agent (agent)      the escalate path: one native tool-calling turn (skipped unless routed escalate)
    guard (python)     assembles the action of the realised path, checks schema and preconditions, gates database
                       writes behind an explicit user confirmation on every path, records the routing, updates memory

Controls: `jev_only_seed()` is the same graph without menu / menu_q / menu_pick / agent, with every step forced
onto the closed path (Jev's route pick is still recorded, so the control measures routing for free);
`llm_only_seed()` is context -> agent, the LLM route as the benchmark's own agent runs it.

`routed_seed(policy=...)` embeds a per-cluster policy from `mixedharness.router.fit_policy` (loop B) as a JSON
literal; the policy is part of the spec and hashes with it.

Observation contract (built by ``examples.tau2.agent.PipelineAgent``):

    obs = {'turn': int, 'policy': str,
           'tools': {name: {'description': str, 'required': [str], 'params': {p: {'type': str, 'description': str}}}},
           'conversation': [msg, ...],   # oldest first, bounded
           'incoming': msg}              # the message that just arrived
    msg = {'role': 'user'|'assistant'|'tool', 'text': str|None, 'tool': str|None,
           'arguments': dict|None, 'result': json|str|None, 'error': bool}

Output contract: {'kind': 'tool', 'name': str, 'arguments': dict} or {'kind': 'message', 'text': str}.
"""
import json

from mixedharness.router import policy_source_literal
from mixedharness.spec import FORMAT, validate_spec

JEV_MODEL = 'jev-1.13.0'

WRITE_TOOLS = ('cancel_pending_order', 'exchange_delivered_order_items', 'modify_pending_order_address',
               'modify_pending_order_items', 'modify_pending_order_payment', 'modify_user_address',
               'return_delivered_order_items')

TRANSFER_TEXT = 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.'
ESCALATE_TAU = 0.6            # Jev's escalate pick needs this route confidence when the compiler has options; else closed
TAU = 0.9                     # accept a closed pick only at this action confidence or above; set from stage 0 (2026-09-25):
                              # on 41 recorded turns every pick at >= 0.9 agreed with the LLM, 6 of 15 below it did
LOW_AGREEMENT_STREAK = 3      # consecutive closed picks with disagreement before trajectory-level escalation
STUCK_REPLIES = 1             # once the same action (tool call or text) has repeated, the next step escalates

# ---------------------------------------------------------------------------------------------------------
# Helpers shared by the python nodes (prepended to each node's source).
# ---------------------------------------------------------------------------------------------------------
COMMON_SOURCE = '''
def clean(token):
    return token.strip(".,;:!?()[]{}\\"'`*")


def tokens_of(text):
    return [clean(t) for t in str(text).split() if clean(t)]


def is_zip(token):
    return len(token) == 5 and token.isdigit()


def is_order_id(token):
    t = token[1:] if token.startswith("#") else token
    return len(t) == 8 and t.startswith("W") and t[1:].isdigit()


def norm_order(token):
    return token if token.startswith("#") else "#" + token


def is_long_id(token):
    return len(token) == 10 and token.isdigit()


def capitalized(token):
    return len(token) > 1 and token[0] >= "A" and token[0] <= "Z" and token[1:].lower() == token[1:] and token.isalpha()


def name_pairs(text):
    stop = set(["I", "My", "Hi", "Hello", "The", "Thanks", "Thank", "Yes", "No", "Please", "Also", "And", "But",
                "Can", "Could", "Would", "It", "This", "That", "Zip", "Email", "Order", "Name", "Sure", "Okay", "Ok",
                "Google", "Apple", "Amazon", "Home", "Kit", "Wifi", "Bluetooth", "Usb"])
    toks = tokens_of(text)
    pairs = []
    for i in range(len(toks) - 1):
        a, b = toks[i], toks[i + 1]
        if capitalized(a) and capitalized(b) and a not in stop and b not in stop:
            pairs.append(a + " " + b)
    return pairs


def yes_no(text):
    t = " " + str(text).lower().strip() + " "
    yes_words = [" yes ", " yes, ", " yes. ", " yeah ", " yep ", " confirm", " go ahead", " proceed", " please do", " that is correct", " that's correct", " correct ", " sounds good", " do it"]
    no_words = [" no ", " no, ", " no. ", " don't ", " do not ", " never mind", " nevermind", " not now", " wait "]
    yes = any(w in t for w in yes_words)
    no = any(w in t for w in no_words)
    if yes and not no:
        return "yes"
    if no and not yes:
        return "no"
    return "unclear"


def intents_of(texts):
    joined = " ".join(str(t).lower() for t in texts)
    flags = {}
    flags["cancel"] = "cancel" in joined
    flags["return"] = "return" in joined
    flags["exchange"] = "exchange" in joined or "swap" in joined
    flags["modify_address"] = ("address" in joined) and ("change" in joined or "update" in joined or "modify" in joined)
    flags["modify_payment"] = ("payment" in joined or "pay with" in joined or "credit card" in joined or "gift card" in joined or "paypal" in joined) and ("change" in joined or "update" in joined or "modify" in joined or "switch" in joined)
    flags["modify_items"] = ("modify" in joined or "change" in joined) and ("item" in joined or "order" in joined) and not flags["modify_address"] and not flags["modify_payment"]
    flags["info"] = "status" in joined or "where is" in joined or "details" in joined or "how much" in joined or "what is" in joined
    flags["human"] = "human" in joined or "representative" in joined or "real person" in joined
    flags["goodbye"] = any(w in joined[-120:] for w in ["thank you", "thanks", "bye", "that's all", "that is all", "nothing else"])
    return flags


def top_intent(intents):
    for key in ["cancel", "return", "exchange", "modify_address", "modify_payment", "modify_items", "human", "info", "goodbye"]:
        if intents.get(key):
            return key
    return "other"


def option_label(item):
    opts = item.get("options") or {}
    parts = [str(k) + "=" + str(v) for k, v in opts.items()]
    return str(item.get("name")) + " [" + ", ".join(parts) + "] price " + str(item.get("price"))


def payment_label(v):
    return str(v.get("source")) + " " + str(v.get("brand") or "") + " " + str(v.get("last_four") or v.get("balance") or "")


def tools_for_agent(tools):
    tools_list = []
    for name, spec in tools.items():
        properties = {}
        for pname, p in (spec.get("params") or {}).items():
            properties[pname] = {"type": p.get("type", "string"), "description": p.get("description", "")}
        tools_list.append({"name": name, "description": spec.get("description", ""),
                           "parameters": {"type": "object", "properties": properties, "required": list(spec.get("required") or [])}})
    return tools_list


def conversation_with_incoming(obs):
    conversation = list(obs.get("conversation") or [])
    if obs.get("incoming"):
        conversation.append(obs["incoming"])
    return conversation


def unit_hash(text):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789 #@.,!?"
    h = 7
    for ch in str(text).lower():
        h = (h * 131 + alphabet.find(ch) + 2) % 1000003
    return h / 1000003.0


def describe_args(args):
    return "; ".join(str(k) + ": " + str(v) for k, v in args.items())


def action_signature(action):
    if not isinstance(action, dict):
        return ""
    if action.get("kind") == "tool":
        args = action.get("arguments") or {}
        return "tool:" + str(action.get("name")) + ":" + ",".join(str(k) + "=" + str(args[k]) for k in sorted(args))
    return "msg:" + str(action.get("text") or "")[:200]


def summarise_order(order):
    items = ", ".join(str(i.get("name")) + " (item " + str(i.get("item_id")) + ", $" + str(i.get("price")) + ")" for i in (order.get("items") or []))
    return "Order " + str(order.get("order_id")) + " is " + str(order.get("status")) + ". Items: " + items + ". Shipping to " + str((order.get("address") or {}).get("city")) + ". How can I help with this order?"


def summarise_product(product):
    avail = [v for v in (product.get("variants") or {}).values() if v.get("available")]
    lines = []
    for v in avail[:12]:
        opts = ", ".join(str(k) + " " + str(val) for k, val in (v.get("options") or {}).items())
        lines.append("item " + str(v.get("item_id")) + ": " + opts + " ($" + str(v.get("price")) + ")")
    return "Available variants of " + str(product.get("name")) + ": " + "; ".join(lines) + ". Which one would you like?"


def summarise_user(user):
    return "You are verified. I can see " + str(len(user.get("orders") or [])) + " orders on your account: " + ", ".join(user.get("orders") or []) + ". Which one can I help with?"


def same_write(pending, name, args):
    if not isinstance(pending, dict) or pending.get("tool") != name:
        return False
    p = pending.get("arguments") or {}
    if "order_id" in p or "order_id" in args:
        if str(p.get("order_id")) != str(args.get("order_id")):
            return False
    if "item_ids" in p or "item_ids" in args:
        if set(str(x) for x in (p.get("item_ids") or [])) != set(str(x) for x in (args.get("item_ids") or [])):
            return False
    return True
'''

# ---------------------------------------------------------------------------------------------------------
# Node 1: compile. The action-space compiler: facts, memory, catalog, argument fillability, cluster, questions.
# ---------------------------------------------------------------------------------------------------------
COMPILE_SOURCE = '''
def run(obs, nodes, memory):
    mem = dict(memory) if isinstance(memory, dict) else {}
    for key, default in [("user_id", None), ("user", None), ("orders", {}), ("products", {}), ("pending", None),
                         ("completed", []), ("transferred", False), ("turns", 0), ("tried", []),
                         ("low_agreement_streak", 0), ("same_reply_streak", 0), ("last_path", None),
                         ("focus_order", None), ("last_signature", None)]:
        mem.setdefault(key, default)
    mem["turns"] = mem["turns"] + 1
    conv = conversation_with_incoming(obs)
    incoming = obs.get("incoming") or {}
    tools = obs.get("tools") or {}
    write_tools = __WRITE_TOOLS__
    user_texts = [m.get("text") or "" for m in conv if m.get("role") == "user"]
    last_user = user_texts[-1] if user_texts else ""
    # --- entities from the tool result that just arrived --------------------------------------------
    if incoming.get("role") == "tool":
        res = incoming.get("result")
        name = incoming.get("tool") or ""
        if incoming.get("error"):
            mem["last_error"] = {"tool": name, "message": str(res)[:300]}
            tried = list(mem["tried"])
            tried.append({"tool": name, "arguments": incoming.get("arguments") or {}})
            mem["tried"] = tried
        else:
            mem["last_error"] = None
            if isinstance(res, str) and name.startswith("find_user_id"):
                mem["user_id"] = res.strip()
            if isinstance(res, dict):
                if "user_id" in res and "orders" in res:
                    mem["user"] = res
                    mem["user_id"] = res.get("user_id")
                if "order_id" in res and "items" in res:
                    orders = dict(mem["orders"])
                    orders[res["order_id"]] = res
                    mem["orders"] = orders
                if "product_id" in res and "variants" in res:
                    products = dict(mem["products"])
                    products[res["product_id"]] = res
                    mem["products"] = products
            if name in write_tools:
                done = list(mem["completed"])
                done.append({"tool": name, "arguments": incoming.get("arguments") or {}})
                mem["completed"] = done
                mem["pending"] = None
            if name == "transfer_to_human_agents":
                mem["transferred"] = True
    else:
        mem["last_error"] = None
    # --- parse user text -----------------------------------------------------------------------------
    all_tokens = []
    for t in user_texts:
        all_tokens.extend(tokens_of(t))
    emails, zips, order_ids, long_ids = [], [], [], []
    for tok in all_tokens:
        if "@" in tok and "." in tok and tok not in emails:
            emails.append(tok)
        elif is_zip(tok) and tok not in zips:
            zips.append(tok)
        elif is_order_id(tok) and norm_order(tok) not in order_ids:
            order_ids.append(norm_order(tok))
        elif is_long_id(tok) and tok not in long_ids:
            long_ids.append(tok)
    names = []
    for t in user_texts:
        for p in name_pairs(t):
            if p not in names:
                names.append(p)
    tried_names = [a["arguments"].get("first_name", "") + " " + a["arguments"].get("last_name", "") for a in mem["tried"] if a.get("tool") == "find_user_id_by_name_zip"]
    tried_emails = [a["arguments"].get("email") for a in mem["tried"] if a.get("tool") == "find_user_id_by_email"]
    names = [n for n in names if n not in tried_names]
    emails = [e for e in emails if e not in tried_emails]
    known_orders = list(mem["orders"].keys())
    user = mem["user"] or {}
    user_orders = list(user.get("orders") or [])
    focus = None
    for oid in reversed(order_ids):
        if oid in user_orders or not user_orders:
            focus = oid
            break
    if focus is None and mem.get("focus_order") in known_orders:
        focus = mem["focus_order"]          # Jev answered "which order" on an earlier turn
    if focus is None and len(known_orders) == 1:
        focus = known_orders[0]             # the one order that has been looked up is the one we are talking about
    if focus is None and len(user_orders) == 1:
        focus = user_orders[0]
    order = mem["orders"].get(focus) if focus else None
    auth = mem["user_id"] is not None
    intents = intents_of(user_texts)
    pending = mem["pending"]
    pm = dict(user.get("payment_methods") or {})
    for entry in ((order or {}).get("payment_history") or []):
        pid = entry.get("payment_method_id") if isinstance(entry, dict) else None
        if pid and pid not in pm:
            pm[pid] = {"source": "order payment", "brand": str(pid).rsplit("_", 1)[0], "last_four": None}
    facts = {
        "authenticated": auth, "user_id": mem["user_id"], "profile_loaded": mem["user"] is not None,
        "emails": emails[:3], "zips": zips[:3], "names": names[:4],
        "order_ids_mentioned": order_ids[:6], "long_ids_mentioned": long_ids[:12], "user_orders": user_orders[:12],
        "orders_loaded": known_orders, "products_loaded": list(mem["products"].keys()),
        "focus_order": focus, "focus_order_loaded": order is not None,
        "focus_order_status": (order or {}).get("status"),
        "last_user_text": last_user[:600], "last_user_says": yes_no(last_user),
        "incoming_role": incoming.get("role"), "incoming_tool": incoming.get("tool"), "incoming_error": bool(incoming.get("error")),
        "last_error": mem.get("last_error"), "intents": intents, "pending": pending, "completed": mem["completed"],
        "transferred": mem["transferred"], "turn": mem["turns"], "user_turns": len(user_texts), "tried": mem["tried"],
        "payment_methods": pm,
    }
    # --- legal-action catalog A(s) ---------------------------------------------------------------------
    options = {}
    if mem["transferred"]:
        options["reply:transfer_notice"] = "Send the mandatory transfer notice; the transfer tool has already been called."
    elif not auth:
        if len(user_texts) == 0 or not last_user.strip():
            options["reply:greeting"] = "Greet the customer and ask how you can help; the agent speaks first and no request has been made yet."
        if emails and "find_user_id_by_email" in tools:
            options["tool:find_user_id_by_email"] = "Authenticate with the email the user gave: " + emails[-1]
        if names and zips and "find_user_id_by_name_zip" in tools:
            options["tool:find_user_id_by_name_zip"] = "Authenticate with the name and zip the user gave (candidates: " + ", ".join(names) + "; zip " + zips[-1] + ")"
        options["reply:ask_identity"] = "Ask the user for their email, or full name plus zip code, to authenticate. Use when nothing usable was given yet or the previous lookup failed."
    else:
        if not facts["profile_loaded"] and "get_user_details" in tools:
            options["tool:get_user_details"] = "Load the authenticated user's profile, orders and payment methods."
        for oid in order_ids:
            if oid not in known_orders and "get_order_details" in tools:
                options["tool:get_order_details:" + oid] = "Load order " + oid + " which the user mentioned."
        if facts["profile_loaded"] and not order_ids and "get_order_details" in tools:
            for oid in user_orders[:6]:
                if oid not in known_orders:
                    options["tool:get_order_details:" + oid] = "Load order " + oid + " from the user's order list (the user did not name an order)."
        if order:
            for it in order.get("items") or []:
                pid = it.get("product_id")
                if pid and pid not in mem["products"] and "get_product_details" in tools:
                    options["tool:get_product_details:" + pid] = "Load variants of " + str(it.get("name")) + " (product " + pid + ") from the focused order; needed before an exchange or item modification."
        if not order:
            for pid in long_ids:
                if pid not in mem["products"] and "get_product_details" in tools:
                    options["tool:get_product_details:" + pid] = "Load product " + pid + " which the user mentioned by id."
        if order and order.get("status") == "pending":
            if intents["cancel"] and "cancel_pending_order" in tools:
                options["tool:cancel_pending_order"] = "Cancel pending order " + focus + ". The harness lists the details and asks for confirmation first."
            if intents["modify_payment"] and "modify_pending_order_payment" in tools and pm:
                options["tool:modify_pending_order_payment"] = "Change the payment method of pending order " + focus + " to one of the user's saved methods."
            if intents["modify_items"] and "modify_pending_order_items" in tools and mem["products"]:
                options["tool:modify_pending_order_items"] = "Modify items of pending order " + focus + " to other variants of the same products (variants loaded)."
        if order and order.get("status") == "delivered":
            if intents["return"] and "return_delivered_order_items" in tools and pm:
                options["tool:return_delivered_order_items"] = "Return items of delivered order " + focus + " to one of the user's payment methods."
            if intents["exchange"] and "exchange_delivered_order_items" in tools and mem["products"]:
                options["tool:exchange_delivered_order_items"] = "Exchange items of delivered order " + focus + " for other variants of the same products (variants loaded)."
        if pending:
            options["reply:confirm_action"] = "Ask the user to confirm the pending " + str(pending.get("tool")) + " action by listing its details; required before any database write."
        if incoming.get("role") == "tool" and not incoming.get("error"):
            options["reply:report_result"] = "Tell the user what the last tool returned (status, items, prices, refund method) in plain words."
        if incoming.get("error"):
            options["reply:report_error"] = "Tell the user the last lookup failed and ask for corrected information."
        options["reply:ask_order"] = "Ask which order the user is referring to (order id), when no order is identified."
        options["reply:deny"] = "Politely decline: the request is outside policy (another user's data, non-pending cancellation, subjective advice, made-up information)."
        if (intents["human"] or (mem.get("last_error") and len(user_texts) > 8)) and "transfer_to_human_agents" in tools:
            options["tool:transfer_to_human_agents"] = "Transfer to a human agent because the request cannot be handled with the available tools."
        if intents["goodbye"] or mem["completed"]:
            options["reply:closing"] = "Close politely; the user's requests have been handled or they said goodbye."
    if not options:
        options["reply:ask_identity"] = "Ask the user for their email, or full name plus zip code."
    # --- argument fillability: what code can fill for every tool option, and what it cannot ------------
    needs = {}
    for oid in options:
        parts = oid.split(":")
        if parts[0] != "tool":
            continue
        tool = parts[1]
        required = list((tools.get(tool) or {}).get("required") or [])
        missing = []
        for arg in required:
            ok = True
            if tool == "find_user_id_by_email":
                ok = len(emails) > 0
            elif tool == "find_user_id_by_name_zip":
                ok = (len(names) > 0) if arg in ("first_name", "last_name") else (len(zips) > 0)
            elif tool == "get_user_details":
                ok = mem["user_id"] is not None
            elif tool == "get_order_details":
                ok = len(parts) > 2 or focus is not None
            elif tool == "get_product_details":
                ok = len(parts) > 2
            elif tool == "transfer_to_human_agents":
                ok = True
            elif arg == "order_id":
                ok = focus is not None
            elif arg == "reason":
                ok = True
            elif arg == "item_ids":
                ok = order is not None and len(order.get("items") or []) > 0
            elif arg == "new_item_ids":
                ok = order is not None and len(mem["products"]) > 0
            elif arg == "payment_method_id":
                ok = len(pm) > 0
            elif arg in ("address1", "city", "state", "country", "zip"):
                ok = False
            if not ok:
                missing.append(arg)
        if missing:
            needs[oid] = missing
    # A step is closed when code can act on it: every listed option is fillable AND the customer's latest message is
    # something the catalog addresses (a tool result to report, no request yet, identity to check, an action intent,
    # an order or product id, an answer to a pending confirmation, a goodbye or a transfer request). A question or a
    # request that maps to nothing in the catalog is open even when every listed option is fillable.
    action_intents = ["cancel", "return", "exchange", "modify_address", "modify_payment", "modify_items"]
    action_signal = (any(intents[k] for k in action_intents) or len(order_ids) > 0 or len(long_ids) > 0
                     or (pending is not None and facts["last_user_says"] != "unclear") or intents["goodbye"] or intents["human"])
    request_covered = incoming.get("role") == "tool" or len(user_texts) == 0 or not last_user.strip() or not auth or action_signal
    closed = len(options) > 0 and len(needs) == 0 and bool(request_covered)
    facts["request_covered"] = bool(request_covered)
    cluster = ("auth" if auth else "anon") + "|" + str(incoming.get("role") or "none") + "|" + str((order or {}).get("status") or "no_order") + "|" + top_intent(intents)
    # --- every question Jev is asked this turn, in one request ----------------------------------------
    questions = {}
    route_criteria = {
        "closed": "One of the listed options is the right next step and every value it needs is already known from the facts. Choose this whenever a listed option fits the customer's latest message, including reporting a tool result or asking a listed question, even if a language model could phrase it better.",
        "escalate": "No listed option does what this turn needs: the customer asked for an explanation, a comparison or a value that is not in the facts, or the next step requires reasoning no option covers. Hand the whole turn to the language-model agent, at the cost of a full model call."}
    if __ALLOW_NARROW__:
        route_criteria["narrow"] = "The request is understood but no listed option fits exactly, or a listed option needs a value that must be composed; a short menu of candidate actions written by a language model, then a pick from it, would settle the turn."
    questions["route"] = {"type": "choice", "instructions": ("You are the routing step of a retail customer-service agent. The facts list what code already knows and "
        "the options list the actions code can execute right now. Judge which path this turn should take; prefer the listed options when one fits."), "criteria": route_criteria}
    questions["action"] = {"type": "choice", "instructions": ("You are the decision step of a retail customer-service agent that must follow the policy in the state. "
        "Choose exactly one next action id. Prefer a tool call when its arguments are already known; before any database write the user must have "
        "explicitly confirmed the listed details; never respond and call a tool in the same turn."), "criteria": options}
    reversed_options = {}
    for oid in reversed(list(options.keys())):
        reversed_options[oid] = options[oid]
    questions["action_alt"] = {"type": "choice", "instructions": ("Independently of any other answer: given the policy, the conversation and the facts, which single option id is the correct "
        "next action for this state? A database write requires an explicit prior confirmation from the user; a tool call must have its arguments known."), "criteria": reversed_options}
    questions["fits"] = {"type": "noul", "instructions": "Does at least one of the listed options fit what the customer's latest message asks for, without needing anything the facts do not contain?"}
    questions["user_confirms"] = {"type": "noul", "instructions": "Does the user's latest message give an explicit yes to proceed with the action the agent just listed? Answer near 0 if the agent has not listed an action yet or the user changed the request."}
    candidates = {}
    if focus is None and len(known_orders) > 1:
        crit = {}
        for oid in known_orders[:8]:
            o = mem["orders"][oid]
            crit[oid] = "Order " + oid + ", " + str(o.get("status")) + ": " + ", ".join(str(i.get("name")) for i in (o.get("items") or [])[:4])
        questions["which_order"] = {"type": "choice", "instructions": "Several of the customer's orders have been looked up. Which one is the customer talking about now?", "criteria": crit}
    if len(names) > 1:
        crit = {}
        for n in names:
            crit[n] = "The customer introduced themselves as " + n
        questions["which_name"] = {"type": "choice", "instructions": "Which of these is the customer's own full name (first and last), rather than another person, a brand or a product?", "criteria": crit}
    if order and order.get("status") == "pending" and intents["cancel"]:
        questions["cancel_reason"] = {"type": "choice", "instructions": "Which reason did the user give for cancelling? Policy allows exactly these two.", "criteria": {"no longer needed": "The user no longer needs the items.", "ordered by mistake": "The user ordered by mistake."}}
    if order and (intents["return"] or intents["exchange"] or intents["modify_items"]):
        verb = "return" if intents["return"] else ("exchange" if intents["exchange"] else "modify")
        for it in order.get("items") or []:
            iid = str(it.get("item_id"))
            questions["item_" + iid] = {"type": "noul", "instructions": "Does the user want to " + verb + " this item: " + option_label(it) + " (item id " + iid + ")?"}
            candidates["item_" + iid] = iid
        if verb != "return":
            for it in order.get("items") or []:
                prod = mem["products"].get(str(it.get("product_id")))
                if not prod:
                    continue
                crit = {}
                for vid, v in list((prod.get("variants") or {}).items())[:30]:
                    if v.get("available") is False:
                        continue
                    crit[str(vid)] = option_label(dict(v, name=prod.get("name")))
                crit[str(it.get("item_id"))] = "keep the current item unchanged"
                if len(crit) > 1:
                    questions["variant_" + str(it.get("item_id"))] = {"type": "choice", "instructions": "If the user wants a different variant of " + str(it.get("name")) + " (currently " + option_label(it) + "), which available variant matches what they asked for? Pick the current item id if they did not ask to change this item.", "criteria": crit}
                    candidates["variant_" + str(it.get("item_id"))] = str(it.get("item_id"))
    if len(pm) > 1 and (intents["return"] or intents["exchange"] or intents["modify_items"] or intents["modify_payment"]):
        crit = {}
        for k, v in pm.items():
            crit[str(k)] = payment_label(v)
        questions["payment_method"] = {"type": "choice", "instructions": "Which of the user's payment methods should receive the refund or be charged, per what the user said? If unspecified, choose the original payment method of the order.", "criteria": crit}
    state = {"policy": obs.get("policy"), "conversation": conv[-12:], "facts": facts,
             "focused_order": order, "payment_methods": pm if pm else None, "options": options, "needs": needs}
    return {"facts": facts, "memory": mem, "options": options, "needs": needs, "closed": closed, "cluster": cluster,
            "questions": questions, "candidates": candidates, "state": state}
'''

# ---------------------------------------------------------------------------------------------------------
# Node 3: route. The routing layer, in order of authority: compiler flag, Jev pick, agreement, policy, trajectory.
# ---------------------------------------------------------------------------------------------------------
ROUTE_SOURCE = '''
def run(obs, nodes, memory):
    cx = nodes["compile"]
    sel = nodes["judge"]
    f = cx["facts"]
    mem = cx["memory"]
    allow_llm = __ALLOW_LLM__
    allow_narrow = __ALLOW_NARROW__
    policy = __POLICY__
    paths = ["closed", "narrow", "escalate"]
    route_answer = sel.get("route") or {}
    path_jev = route_answer.get("choice") if route_answer.get("choice") in paths else "closed"
    route_confidence = (route_answer.get("probabilities") or {}).get(path_jev)
    action_answer = sel.get("action") or {}
    chosen = action_answer.get("choice")
    confidence = (action_answer.get("probabilities") or {}).get(chosen)
    alt = (sel.get("action_alt") or {}).get("choice")
    agreement = None if (chosen is None or alt is None) else (chosen == alt)
    fits = (sel.get("fits") or {}).get("noul")
    path = path_jev
    source = "jev"
    escalate_confidence = (route_answer.get("probabilities") or {}).get("escalate")
    if path == "escalate" and cx["closed"] and (escalate_confidence is None or escalate_confidence < __ESCALATE_TAU__):
        path, source = "closed", "escalate_below_tau"   # symmetric threshold: an unsure escalate stays on the listed options
    if cx["closed"] and f["incoming_role"] == "tool" and f["pending"] is None and path != "closed":
        path, source = "closed", "tool_result"          # a tool result with listed options is reported by code, not by the LLM
    if not cx["closed"] and path == "closed":
        path, source = "narrow", "code_override"
    if path == "closed" and (confidence is None or confidence < __TAU__):
        path, source = "escalate", "low_confidence"
    if path == "closed" and agreement is False:
        path, source = "narrow", "fallthrough"
    prefer = (policy.get("prefer") or {}).get(cx["cluster"])
    if prefer in paths and prefer != path:
        path, source = prefer, "policy"
    epsilon = float(policy.get("epsilon") or 0.0)
    shadow_of = None
    if epsilon > 0.0 and unit_hash(str(f["turn"]) + "|" + f["last_user_text"] + "|" + cx["cluster"]) < epsilon:
        shadow_of = path
        path = paths[(paths.index(path) + 1) % len(paths)]
        source = "shadow"
    if not cx["closed"] and path == "closed":
        path = "narrow"   # neither the policy nor a shadow draw may put an open step on the closed path
    if mem.get("low_agreement_streak", 0) >= __LOW_AGREEMENT_STREAK__ or mem.get("same_reply_streak", 0) >= __STUCK_REPLIES__:
        if path != "escalate" and allow_llm:
            path, source = "escalate", "trajectory"
    if not allow_narrow and path == "narrow":
        path = "escalate"   # the narrow path is off: the LLM takes the step itself (source unchanged)
    if not allow_llm and path != "closed":
        path, source = "closed", "forced"
    if mem.get("transferred") and path != "closed":
        path, source = "closed", "forced"
    menu_prompt = None
    agent_context = None
    if path == "escalate":
        agent_context = {"policy": obs.get("policy"), "tools": tools_for_agent(obs.get("tools") or {}), "conversation": conversation_with_incoming(obs)}
    if path == "narrow":
        tools = obs.get("tools") or {}
        tool_lines = []
        for name, spec in tools.items():
            params = ", ".join(p + " (" + str(v.get("type", "string")) + ")" for p, v in (spec.get("params") or {}).items())
            tool_lines.append("- " + name + "(" + params + "): " + str(spec.get("description", ""))[:200])
        conv_lines = []
        for m in conversation_with_incoming(obs)[-12:]:
            if m.get("role") == "tool":
                conv_lines.append("tool " + str(m.get("tool")) + " -> " + str(m.get("result"))[:600])
            elif m.get("tool"):
                conv_lines.append("assistant called " + str(m.get("tool")) + " " + str(m.get("arguments")))
            else:
                conv_lines.append(str(m.get("role")) + ": " + str(m.get("text"))[:600])
        known = {"user_id": f["user_id"], "focus_order": f["focus_order"], "focus_order_status": f["focus_order_status"],
                 "orders_loaded": f["orders_loaded"], "products_loaded": f["products_loaded"], "pending_write": f["pending"],
                 "completed_writes": f["completed"], "options_code_can_execute": list(cx["options"].keys()),
                 "note": "orders_loaded and products_loaded are already in memory: never propose fetching them again; use their contents"}
        menu_prompt = ("You are a retail customer-service agent. Follow this policy exactly:\\n" + str(obs.get("policy") or "") +
                       "\\n\\nTools:\\n" + "\\n".join(tool_lines) +
                       "\\n\\nWhat the system already knows: " + str(known) +
                       "\\n\\nConversation (oldest first):\\n" + "\\n".join(conv_lines) +
                       "\\n\\nWrite a short menu of 2 to 4 candidate next actions for the agent, each different, each allowed by the policy. "
                       "A candidate is either one tool call with complete arguments, or one message to the customer of at most 80 words. "
                       "Never propose a database write unless the customer has explicitly confirmed its exact details; propose asking for confirmation instead. "
                       "Give each candidate a short id (a, b, c, d) and one line saying when it is the right choice.")
    return {"path": path, "path_jev": path_jev, "path_source": source, "shadow_of": shadow_of, "route_confidence": route_confidence,
            "agreement": agreement, "confidence": confidence, "fits": fits, "cluster": cx["cluster"], "closed": cx["closed"],
            "needs": cx["needs"], "menu_prompt": menu_prompt, "agent_context": agent_context}
'''

# ---------------------------------------------------------------------------------------------------------
# Node 4: decision. The closed path: Jev's answers -> a tool call or a template message.
# ---------------------------------------------------------------------------------------------------------
DECISION_SOURCE = '''
def run(obs, nodes, memory):
    cx = nodes["compile"]
    f = cx["facts"]
    mem = dict(cx["memory"])
    sel = nodes["judge"]
    tools = obs.get("tools") or {}
    write_tools = __WRITE_TOOLS__
    allow_llm = __ALLOW_LLM__
    focus = f["focus_order"]
    order = (mem.get("orders") or {}).get(focus) if focus else None
    pm = f.get("payment_methods") or {}
    chosen = (sel.get("action") or {}).get("choice")
    confirms = float((sel.get("user_confirms") or {}).get("noul") or 0.0)
    which = (sel.get("which_order") or {}).get("choice")
    if which and which in (mem.get("orders") or {}):
        mem["focus_order"] = which
    action = None
    reply = None
    tool = None
    args = {}
    if chosen is None or chosen not in cx["options"]:
        reply = "ask_identity" if not f["authenticated"] else "ask_order"
    else:
        parts = chosen.split(":")
        if parts[0] == "reply":
            reply = parts[1]
        else:
            tool = parts[1]
    if tool is not None:
        parts = chosen.split(":")
        if tool == "find_user_id_by_email":
            args["email"] = f["emails"][-1]
        elif tool == "find_user_id_by_name_zip":
            name = (sel.get("which_name") or {}).get("choice") or f["names"][-1]
            fl = name.split(" ")
            args["first_name"] = fl[0]
            args["last_name"] = fl[-1]
            args["zip"] = f["zips"][-1]
        elif tool == "get_user_details":
            args["user_id"] = f["user_id"]
        elif tool == "get_order_details":
            args["order_id"] = parts[2] if len(parts) > 2 else focus
        elif tool == "get_product_details":
            args["product_id"] = parts[2]
        elif tool == "transfer_to_human_agents":
            args["summary"] = "User request cannot be handled with the available tools. Last message: " + f["last_user_text"][:200]
        elif tool == "cancel_pending_order":
            args["order_id"] = focus
            args["reason"] = (sel.get("cancel_reason") or {}).get("choice") or "no longer needed"
        elif tool in ("return_delivered_order_items", "exchange_delivered_order_items", "modify_pending_order_items"):
            args["order_id"] = focus
            item_ids = []
            for key, iid in (cx.get("candidates") or {}).items():
                if key.startswith("item_") and float((sel.get(key) or {}).get("noul") or 0.0) > 0.5:
                    item_ids.append(iid)
            args["item_ids"] = item_ids
            if tool != "return_delivered_order_items":
                new_ids = []
                for iid in item_ids:
                    new_ids.append(((sel.get("variant_" + iid) or {}).get("choice")) or iid)
                args["new_item_ids"] = new_ids
            if "payment_method" in sel:
                args["payment_method_id"] = sel["payment_method"].get("choice")
            elif len(pm) == 1:
                args["payment_method_id"] = list(pm.keys())[0]
        elif tool == "modify_pending_order_payment":
            args["order_id"] = focus
            if "payment_method" in sel:
                args["payment_method_id"] = sel["payment_method"].get("choice")
            elif len(pm) == 1:
                args["payment_method_id"] = list(pm.keys())[0]
        required = (tools.get(tool) or {}).get("required") or []
        missing = [r for r in required if r not in args or args.get(r) in (None, "", [])]
        if missing:
            mem["last_error"] = {"tool": tool, "message": "missing arguments " + ", ".join(missing)}
            if "item_ids" in missing or "new_item_ids" in missing:
                reply = "ask_items"
            elif "order_id" in missing:
                reply = "ask_order"
            else:
                reply = "report_error"
            tool = None
    # Without a language model, a stuck conversation is transferred; with one, the route node escalates it first.
    if not allow_llm and tool is None and reply not in (None, "transfer_notice", "closing") and mem.get("same_reply_streak", 0) >= __STUCK_REPLIES__ and "transfer_to_human_agents" in tools and not mem.get("transferred"):
        tool = "transfer_to_human_agents"
        args = {"summary": "Conversation is stuck; the agent could not identify the requested action from the user's messages. Last message: " + f["last_user_text"][:200]}
        reply = None
    if mem.get("pending") is not None and f["last_user_says"] == "no" and f["incoming_role"] == "user":
        mem["pending"] = None
        tool = None
        reply = "ask_change"
    pending_now = mem.get("pending")
    if pending_now is not None and f["incoming_role"] == "user" and (f["last_user_says"] == "yes" or confirms > 0.5) and f["last_user_says"] != "no" \
            and pending_now.get("tool") in tools and tool != pending_now.get("tool"):
        # The customer just confirmed the listed write (set by any path); execute exactly what was listed.
        tool = pending_now["tool"]
        args = dict(pending_now.get("arguments") or {})
        reply = None
    if tool is not None:
        if tool in write_tools:
            pending = mem.get("pending")
            same = pending is not None and pending.get("tool") == tool and pending.get("arguments") == args
            if same and confirms > 0.5 and f["last_user_says"] != "no":
                action = {"kind": "tool", "name": tool, "arguments": args}
            else:
                mem["pending"] = {"tool": tool, "arguments": args}
                reply = "confirm_action"
        else:
            action = {"kind": "tool", "name": tool, "arguments": args}
    if action is None:
        text = None
        if reply == "transfer_notice":
            text = __TRANSFER_TEXT__
            mem["transferred"] = False
        elif reply == "greeting":
            text = "Hi! How can I help you today?"
        elif reply == "ask_identity":
            text = "To get started, could you please provide the email address on your account, or your full name together with your zip code, so I can verify your identity?"
        elif reply == "ask_order":
            text = "Could you tell me the order id you are referring to (it starts with #W)? If you are not sure, I can look through your orders."
        elif reply == "ask_change":
            text = "Understood, I will not proceed with that. What would you like to change, or is there something else I can help with?"
        elif reply == "ask_items":
            items = ", ".join(str(i.get("name")) + " (item " + str(i.get("item_id")) + ")" for i in ((order or {}).get("items") or []))
            text = "Which of these items would you like to change, and to which options? " + items + ". Please name the item and the option you want, and I will check availability."
        elif reply == "confirm_action":
            p = mem.get("pending") or {}
            text = ("Please confirm the following action so I can proceed: " + str(p.get("tool")) + " with " + describe_args(p.get("arguments") or {}) +
                    ". Reply 'yes' to confirm or tell me what to change.")
        elif reply == "report_result":
            res = (obs.get("incoming") or {}).get("result")
            if isinstance(res, dict) and "order_id" in res:
                items = res.get("items") or []
                names = ", ".join(str(i.get("name")) + " (item " + str(i.get("item_id")) + ", $" + str(i.get("price")) + ")" for i in items)
                text = "Order " + str(res.get("order_id")) + " is " + str(res.get("status")) + ". Items: " + names + ". Shipping to " + str((res.get("address") or {}).get("city")) + ". How can I help with this order?"
            elif isinstance(res, dict) and "user_id" in res:
                text = "Thank you, you are verified. I can see " + str(len(res.get("orders") or [])) + " orders on your account: " + ", ".join(res.get("orders") or []) + ". Which one can I help with?"
            elif isinstance(res, dict) and "product_id" in res:
                avail = [v for v in (res.get("variants") or {}).values() if v.get("available")]
                text = "For " + str(res.get("name")) + " there are " + str(len(avail)) + " available variants. Tell me which options you want and I will list the matching item."
            elif isinstance(res, dict) and "status" in res:
                text = "Done. The order is now " + str(res.get("status")) + ". Is there anything else I can help with?"
            else:
                text = "Here is what I found: " + str(res)[:400] + ". How would you like to proceed?"
        elif reply == "report_error":
            text = "I could not complete that lookup (" + str((mem.get("last_error") or {}).get("message"))[:120] + "). Could you double-check the information and send it again?"
        elif reply == "deny":
            text = "I'm sorry, but I cannot help with that request under our policy. Is there something else I can do for you?"
        elif reply == "closing":
            text = "You're welcome. Your request has been taken care of. Have a great day!"
        else:
            text = "Could you tell me a bit more about what you would like to do with your order?"
        action = {"kind": "message", "text": text}
    return {"action": action, "memory": mem, "chosen": chosen, "reply": reply, "confirms": confirms}
'''

# ---------------------------------------------------------------------------------------------------------
# Node 6: menu_q. The narrow path's menu as a Jev choice question; null when there is no menu this turn.
# ---------------------------------------------------------------------------------------------------------
MENU_Q_SOURCE = '''
def run(obs, nodes, memory):
    menu = nodes.get("menu")
    cx = nodes["compile"]
    if not isinstance(menu, dict) or not isinstance(menu.get("options"), list) or not menu["options"]:
        return {"questions": None, "state": None, "options": {}}
    crit = {}
    by_id = {}
    for i, opt in enumerate(menu["options"][:4]):
        if not isinstance(opt, dict):
            continue
        oid = str(opt.get("id") or "abcd"[i])
        if opt.get("kind") == "tool" and opt.get("tool"):
            crit[oid] = "Call " + str(opt["tool"]) + " with " + str(opt.get("arguments") or {}) + ". When: " + str(opt.get("when") or "")[:200]
        else:
            crit[oid] = "Reply: " + str(opt.get("text") or "")[:300] + " When: " + str(opt.get("when") or "")[:200]
        by_id[oid] = opt
    if not crit:
        return {"questions": None, "state": None, "options": {}}
    questions = {"pick": {"type": "choice", "instructions": ("Pick the single best next action for this retail customer-service turn, following the policy in the state. "
                 "A database write needs an explicit prior confirmation of its exact details by the user."), "criteria": crit}}
    state = {"policy": obs.get("policy"), "conversation": conversation_with_incoming(obs)[-12:], "facts": cx["facts"]}
    return {"questions": questions, "state": state, "options": by_id}
'''

# ---------------------------------------------------------------------------------------------------------
# Node 9: guard. Assemble the realised path's action, check it, gate writes, record the routing, update memory.
# ---------------------------------------------------------------------------------------------------------
GUARD_SOURCE = '''
def run(obs, nodes, memory):
    cx = nodes["compile"]
    rt = nodes["route"]
    dec = nodes["decision"]
    f = cx["facts"]
    tools = obs.get("tools") or {}
    write_tools = __WRITE_TOOLS__
    path = rt["path"]
    source = rt["path_source"]
    mem = dict(dec["memory"]) if path == "closed" else dict(cx["memory"])
    action = None
    chosen = dec.get("chosen")
    guard = "ok"
    if path == "narrow":
        pick = nodes.get("menu_pick") or {}
        mq = nodes.get("menu_q") or {}
        oid = ((pick.get("pick") or {}).get("choice")) if isinstance(pick, dict) else None
        opt = (mq.get("options") or {}).get(oid) if oid else None
        if isinstance(opt, dict):
            chosen = "menu:" + str(oid)
            if opt.get("kind") == "tool" and opt.get("tool"):
                action = {"kind": "tool", "name": str(opt["tool"]), "arguments": dict(opt.get("arguments") or {})}
            elif isinstance(opt.get("text"), str) and opt["text"].strip():
                action = {"kind": "message", "text": opt["text"].strip()}
        if action is None:
            path, source, guard = "closed", "guard", "menu_invalid"
            mem = dict(dec["memory"])
    elif path == "escalate":
        ag = nodes.get("agent")
        if isinstance(ag, dict) and ag.get("kind") == "tool" and ag.get("name"):
            action = {"kind": "tool", "name": str(ag["name"]), "arguments": dict(ag.get("arguments") or {})}
            chosen = "agent:tool:" + str(ag["name"])
        elif isinstance(ag, dict) and isinstance(ag.get("text"), str) and ag["text"].strip():
            action = {"kind": "message", "text": ag["text"].strip()}
            chosen = "agent:message"
        if action is None:
            path, source, guard = "closed", "guard", "agent_invalid"
            mem = dict(dec["memory"])
    if action is None:
        action = dec["action"]
    # --- schema and precondition checks on tool calls from any path -----------------------------------
    if action.get("kind") == "tool":
        name = action.get("name")
        spec = tools.get(name)
        args = action.get("arguments") or {}
        if spec is None:
            guard = "unknown_tool"
            action = dec["action"] if dec["action"].get("kind") == "message" else {"kind": "message", "text": "Could you tell me a bit more about what you would like to do with your order?"}
            path, source = "closed", "guard"
            mem = dict(dec["memory"])
        else:
            missing = [r for r in (spec.get("required") or []) if r not in args or args.get(r) in (None, "", [])]
            unknown = [k for k in args if k not in (spec.get("params") or {})]
            if missing or unknown:
                guard = "bad_arguments"
                action = {"kind": "message", "text": "I need a little more information before I can do that. Could you confirm the details (order id, items and the option you want)?"}
            elif name in write_tools and path != "closed":
                # The irreversible gate on the LLM paths: the customer said yes to this tool on this order (item set
                # included when there is one); other argument differences between turns are not a new operation.
                pending = mem.get("pending")
                same = same_write(pending, name, args)
                confirmed = f["last_user_says"] == "yes" or (float(dec.get("confirms") or 0.0) > 0.5 and f["last_user_says"] != "no")
                if not (same and confirmed):
                    mem["pending"] = {"tool": name, "arguments": args}
                    guard = "write_gate"
                    action = {"kind": "message", "text": "Please confirm the following action so I can proceed: " + str(name) + " with " + describe_args(args) + ". Reply 'yes' to confirm or tell me what to change."}
    # --- a read already in memory is answered from memory, not re-fetched ------------------------------
    if action.get("kind") == "tool" and guard == "ok":
        name = action.get("name")
        args = action.get("arguments") or {}
        if name == "get_order_details" and str(args.get("order_id")) in (mem.get("orders") or {}):
            action = {"kind": "message", "text": summarise_order(mem["orders"][str(args["order_id"])])}
            guard = "memory_answer"
            mem["focus_order"] = str(args["order_id"])
        elif name == "get_product_details" and str(args.get("product_id")) in (mem.get("products") or {}):
            action = {"kind": "message", "text": summarise_product(mem["products"][str(args["product_id"])])}
            guard = "memory_answer"
        elif name == "get_user_details" and mem.get("user") and str(args.get("user_id")) == str(mem.get("user_id")):
            action = {"kind": "message", "text": summarise_user(mem["user"])}
            guard = "memory_answer"
    # --- trajectory counters --------------------------------------------------------------------------
    if rt["path_jev"] == "closed" and rt["agreement"] is False:
        mem["low_agreement_streak"] = mem.get("low_agreement_streak", 0) + 1
    else:
        mem["low_agreement_streak"] = 0
    signature = action_signature(action)
    if signature and signature == mem.get("last_signature"):
        mem["same_reply_streak"] = mem.get("same_reply_streak", 0) + 1   # the same tool call or the same text, twice in a row
    else:
        mem["same_reply_streak"] = 0
    mem["last_signature"] = signature
    if path == "escalate":
        mem["low_agreement_streak"] = 0
        mem["same_reply_streak"] = 0
    mem["last_path"] = path
    routing = {"cluster": rt["cluster"], "closed": rt["closed"], "needs": rt["needs"], "path_jev": rt["path_jev"], "path": path,
               "path_source": source, "shadow_of": rt.get("shadow_of"), "agreement": rt["agreement"], "confidence": rt["confidence"],
               "route_confidence": rt.get("route_confidence"), "fits": rt["fits"]}
    return {"action": action, "memory": mem, "chosen": chosen, "guard": guard, "routing": routing}
'''

AGENT_INSTRUCTIONS = ('You are a retail customer-service agent. Follow the policy exactly. In each turn either call exactly one '
                      'tool or send one message to the customer, never both. Before any database write, list the details and get '
                      'an explicit yes. Never invent order or product details; look them up.')

CONTEXT_SOURCE = COMMON_SOURCE + '''
def run(obs, nodes, memory):
    return {"policy": obs.get("policy"), "tools": tools_for_agent(obs.get("tools") or {}), "conversation": conversation_with_incoming(obs)}
'''

MENU_SCHEMA = {'type': 'object', 'required': ['options'], 'additionalProperties': False,
               'properties': {'options': {'type': 'array', 'minItems': 1, 'maxItems': 4, 'items': {
                   'type': 'object', 'required': ['id', 'kind'], 'additionalProperties': False,
                   'properties': {'id': {'type': 'string'}, 'kind': {'type': 'string', 'enum': ['tool', 'message']},
                                  'tool': {'type': 'string'}, 'arguments': {'type': 'object'}, 'text': {'type': 'string'},
                                  'when': {'type': 'string'}}}}}}


def _fill(source, *, allow_llm, policy=None, tau=TAU, narrow=False):
    return (COMMON_SOURCE + source).replace('__ALLOW_LLM__', 'True' if allow_llm else 'False') \
        .replace('__ALLOW_NARROW__', 'True' if narrow else 'False') \
        .replace('__TAU__', repr(float(tau))) \
        .replace('__ESCALATE_TAU__', repr(float(ESCALATE_TAU))) \
        .replace('__WRITE_TOOLS__', json.dumps(list(WRITE_TOOLS))) \
        .replace('__TRANSFER_TEXT__', json.dumps(TRANSFER_TEXT)) \
        .replace('__POLICY__', policy_source_literal(policy)) \
        .replace('__LOW_AGREEMENT_STREAK__', str(LOW_AGREEMENT_STREAK)) \
        .replace('__STUCK_REPLIES__', str(STUCK_REPLIES))


def _core_nodes(*, allow_llm, policy=None, tau=TAU, narrow=False):
    return [
        {'id': 'compile', 'kind': 'python', 'source': _fill(COMPILE_SOURCE, allow_llm=allow_llm, narrow=narrow), 'inputs': []},
        {'id': 'judge', 'kind': 'jev', 'state': "nodes['compile']['state']", 'questions_expression': "nodes['compile']['questions']"},
        {'id': 'route', 'kind': 'python', 'source': _fill(ROUTE_SOURCE, allow_llm=allow_llm, policy=policy, tau=tau, narrow=narrow), 'inputs': ['compile', 'judge']},
        {'id': 'decision', 'kind': 'python', 'source': _fill(DECISION_SOURCE, allow_llm=allow_llm), 'inputs': ['compile', 'judge']},
    ]


def jev_only_seed():
    """Every step on the closed path; Jev's route pick and the agreement signal are recorded but never acted on."""
    nodes = _core_nodes(allow_llm=False) + [
        {'id': 'guard', 'kind': 'python', 'source': _fill(GUARD_SOURCE, allow_llm=False), 'inputs': ['compile', 'route', 'decision']}]
    spec = {'format': FORMAT, 'name': 'tau2 retail · jev only', 'jev_model': JEV_MODEL, 'nodes': nodes,
            'output': "nodes['guard']['action']", 'memory': "nodes['guard']['memory']"}
    return validate_spec(spec)


def routed_seed(llm_model='claude-sonnet-5', *, policy=None, tau=TAU, narrow=False):
    """The routed harness: compiler, one Jev request, the paths, guard. `policy` is a fitted loop-B policy or None;
    `tau` the action-confidence threshold below which a closed pick escalates; `narrow` turns the menu path on
    (off by default since stage 2: it cost more than escalate and produced the loops)."""
    nodes = _core_nodes(allow_llm=True, policy=policy, tau=tau, narrow=narrow) + [
        {'id': 'menu', 'kind': 'llm', 'prompt': "nodes['route']['menu_prompt']", 'schema': MENU_SCHEMA, 'max_tokens': 2000, 'on_error': 'null'},
        {'id': 'menu_q', 'kind': 'python', 'source': _fill(MENU_Q_SOURCE, allow_llm=True), 'inputs': ['compile', 'menu']},
        {'id': 'menu_pick', 'kind': 'jev', 'state': "nodes['menu_q']['state']", 'questions_expression': "nodes['menu_q']['questions']"},
        {'id': 'agent', 'kind': 'agent', 'instructions': AGENT_INSTRUCTIONS, 'context': "nodes['route']['agent_context']", 'max_tokens': 1024},
        {'id': 'guard', 'kind': 'python', 'source': _fill(GUARD_SOURCE, allow_llm=True),
         'inputs': ['compile', 'route', 'decision', 'menu_q', 'menu_pick', 'agent']}]
    spec = {'format': FORMAT, 'name': 'tau2 retail · routed', 'jev_model': JEV_MODEL, 'llm_model': llm_model, 'nodes': nodes,
            'output': "nodes['guard']['action']", 'memory': "nodes['guard']['memory']"}
    return validate_spec(spec)


def llm_only_seed(llm_model='claude-sonnet-5'):
    """Every step escalates: the LLM route as the benchmark's own agent runs it, through our runtime."""
    nodes = [{'id': 'context', 'kind': 'python', 'source': CONTEXT_SOURCE, 'inputs': []},
             {'id': 'agent', 'kind': 'agent', 'instructions': AGENT_INSTRUCTIONS, 'context': "nodes['context']", 'max_tokens': 1024}]
    spec = {'format': FORMAT, 'name': 'tau2 retail · llm only', 'jev_model': JEV_MODEL, 'llm_model': llm_model,
            'nodes': nodes, 'output': "nodes['agent']"}
    return validate_spec(spec)


SEEDS = {'jev_only': jev_only_seed, 'routed': routed_seed, 'llm_only': llm_only_seed}
