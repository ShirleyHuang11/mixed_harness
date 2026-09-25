"""Seed harnesses for the tau2-bench retail domain.

Two seeds share every node except the ``llm`` nodes:

* ``jev_only_seed()`` – code builds the legal-action catalog and every candidate question, Jev answers them
  all in one request (speculative fan-out: the action choice plus the selections the action might need),
  code turns the answers into a tool call or a template message. No language model anywhere.
* ``harness_seed()``  – the same graph plus ``draft_reply`` and ``draft_args`` llm nodes. Their prompts are
  ``null`` unless the decision node asks for generation, and a null prompt skips the node (runtime rule).

Graph (one turn):  facts (python) -> judge (jev) -> decision (python) [-> draft_reply / draft_args (llm) -> final (python)]

Sandboxed Python nodes cost a fixed process spawn each (~0.5 s on this cluster, measured), so the graph is kept
to two python nodes for jev_only and three for harness; the ledger reports model-only latency separately.

Observation contract (built by ``examples.tau2.agent.PipelineAgent``):

    obs = {'turn': int, 'policy': str,
           'tools': {name: {'description': str, 'required': [str], 'params': {p: {'type': str, 'description': str}}}},
           'conversation': [msg, ...],   # oldest first, bounded
           'incoming': msg}              # the message that just arrived
    msg = {'role': 'user'|'assistant'|'tool', 'text': str|None, 'tool': str|None,
           'arguments': dict|None, 'result': json|str|None, 'error': bool}

Output contract: {'kind': 'tool', 'name': str, 'arguments': dict} or {'kind': 'message', 'text': str}.
"""
import copy
import json

from mixedharness.spec import FORMAT, validate_spec

JEV_MODEL = 'jev-1.13.0'

WRITE_TOOLS = ('cancel_pending_order', 'exchange_delivered_order_items', 'modify_pending_order_address',
               'modify_pending_order_items', 'modify_pending_order_payment', 'modify_user_address',
               'return_delivered_order_items')

TRANSFER_TEXT = 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.'

# ---------------------------------------------------------------------------------------------------------
# Node 1: facts. Parse the conversation, keep fetched entities, build the catalog and every question.
# ---------------------------------------------------------------------------------------------------------
FACTS_SOURCE = '''
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
    no_words = [" no ", " no, ", " no. ", " don't ", " do not ", " cancel that", " never mind", " nevermind", " not now", " wait "]
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


def option_label(item):
    opts = item.get("options") or {}
    parts = [str(k) + "=" + str(v) for k, v in opts.items()]
    return str(item.get("name")) + " [" + ", ".join(parts) + "] price " + str(item.get("price"))


def payment_label(v):
    return str(v.get("source")) + " " + str(v.get("brand") or "") + " " + str(v.get("last_four") or v.get("balance") or "")


def run(obs, nodes, memory):
    mem = dict(memory) if isinstance(memory, dict) else {}
    for key, default in [("user_id", None), ("user", None), ("orders", {}), ("products", {}), ("pending", None),
                         ("completed", []), ("transferred", False), ("turns", 0), ("tried", [])]:
        mem.setdefault(key, default)
    mem["turns"] = mem["turns"] + 1
    conv = list(obs.get("conversation") or [])
    incoming = obs.get("incoming") or {}
    if incoming:
        conv = conv + [incoming]
    tools = obs.get("tools") or {}
    allow_llm = __ALLOW_LLM__
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
    if focus is None and len(user_orders) == 1:
        focus = user_orders[0]
    order = mem["orders"].get(focus) if focus else None
    auth = mem["user_id"] is not None
    intents = intents_of(user_texts)
    pending = mem["pending"]
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
    }
    # --- legal-action catalog ------------------------------------------------------------------------
    options = {}
    if mem["transferred"]:
        options["reply:transfer_notice"] = "Send the mandatory transfer notice; the transfer tool has already been called."
    elif not auth:
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
            if intents["modify_payment"] and "modify_pending_order_payment" in tools and facts["profile_loaded"]:
                options["tool:modify_pending_order_payment"] = "Change the payment method of pending order " + focus + " to one of the user's saved methods."
            if intents["modify_items"] and "modify_pending_order_items" in tools and mem["products"]:
                options["tool:modify_pending_order_items"] = "Modify items of pending order " + focus + " to other variants of the same products (variants loaded)."
            if intents["modify_address"] and "modify_pending_order_address" in tools and allow_llm:
                options["tool:modify_pending_order_address"] = "Change the shipping address of pending order " + focus + " to the address the user wrote."
        if order and order.get("status") == "delivered":
            if intents["return"] and "return_delivered_order_items" in tools and facts["profile_loaded"]:
                options["tool:return_delivered_order_items"] = "Return items of delivered order " + focus + " to one of the user's payment methods."
            if intents["exchange"] and "exchange_delivered_order_items" in tools and mem["products"]:
                options["tool:exchange_delivered_order_items"] = "Exchange items of delivered order " + focus + " for other variants of the same products (variants loaded)."
        if intents["modify_address"] and not order and "modify_user_address" in tools and allow_llm and facts["profile_loaded"]:
            options["tool:modify_user_address"] = "Change the user's default address to the address they wrote."
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
        if allow_llm:
            options["reply:llm"] = "Write a free-form reply (explanations, multi-part answers, clarifying questions no template covers). Costs a language-model call; use only when no template above fits."
    if not options:
        options["reply:ask_identity"] = "Ask the user for their email, or full name plus zip code."
    # --- every question Jev might need this turn, asked in one request -------------------------------
    questions = {}
    questions["action"] = {"type": "choice", "instructions": ("You are the decision step of a retail customer-service agent that must follow the policy in the state. "
        "Choose exactly one next action id. Prefer a tool call when its arguments are already known; before any database write the user must have "
        "explicitly confirmed the listed details; never respond and call a tool in the same turn."), "criteria": options}
    questions["user_confirms"] = {"type": "noul", "instructions": "Does the user's latest message give an explicit yes to proceed with the action the agent just listed? Answer near 0 if the agent has not listed an action yet or the user changed the request."}
    candidates = {}
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
    pm = user.get("payment_methods") or {}
    if len(pm) > 1 and (intents["return"] or intents["exchange"] or intents["modify_items"] or intents["modify_payment"]):
        crit = {}
        for k, v in pm.items():
            crit[str(k)] = payment_label(v)
        questions["payment_method"] = {"type": "choice", "instructions": "Which of the user's payment methods should receive the refund or be charged, per what the user said? If unspecified, choose the original payment method of the order.", "criteria": crit}
    state = {"policy": obs.get("policy"), "conversation": conv[-12:], "facts": facts,
             "focused_order": order, "payment_methods": pm if pm else None}
    return {"facts": facts, "memory": mem, "options": options, "questions": questions, "candidates": candidates, "state": state}
'''

# ---------------------------------------------------------------------------------------------------------
# Node 2: decision. Turn Jev's answers into a tool call or a template message; ask for generation if needed.
# ---------------------------------------------------------------------------------------------------------
DECISION_SOURCE = '''
def describe_args(args):
    return "; ".join(str(k) + ": " + str(v) for k, v in args.items())


def run(obs, nodes, memory):
    fx = nodes["facts"]
    f = fx["facts"]
    mem = dict(fx["memory"])
    sel = nodes["judge"]
    tools = obs.get("tools") or {}
    allow_llm = __ALLOW_LLM__
    write_tools = __WRITE_TOOLS__
    focus = f["focus_order"]
    order = (mem.get("orders") or {}).get(focus) if focus else None
    user = mem.get("user") or {}
    pm = user.get("payment_methods") or {}
    chosen = (sel.get("action") or {}).get("choice")
    confidence = ((sel.get("action") or {}).get("probabilities") or {}).get(chosen)
    confirms = float((sel.get("user_confirms") or {}).get("noul") or 0.0)
    llm_prompt = None
    llm_args_prompt = None
    action = None
    reply = None
    tool = None
    args = {}
    if chosen is None or chosen not in fx["options"]:
        reply = "ask_identity" if not f["authenticated"] else "ask_order"
    else:
        parts = chosen.split(":")
        if parts[0] == "reply":
            reply = parts[1]
        else:
            tool = parts[1]
    if tool is not None:
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
            for key, iid in (fx.get("candidates") or {}).items():
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
        elif tool in ("modify_pending_order_address", "modify_user_address"):
            if tool == "modify_pending_order_address":
                args["order_id"] = focus
            else:
                args["user_id"] = f["user_id"]
            if allow_llm:
                spec = tools.get(tool) or {}
                llm_args_prompt = ("Extract the tool arguments from the conversation. Tool: " + tool + ". Parameters: " + str(spec.get("params")) +
                                   ". Already known: " + str(args) + ". Conversation (latest last): " + str((obs.get("conversation") or [])[-8:]) +
                                   ". Return every required parameter; do not invent values the user did not state.")
        required = (tools.get(tool) or {}).get("required") or []
        missing = [r for r in required if r not in args or args.get(r) in (None, "", [])]
        if missing and llm_args_prompt is None:
            mem["last_error"] = {"tool": tool, "message": "missing arguments " + ", ".join(missing)}
            if "item_ids" in missing or "new_item_ids" in missing:
                reply = "ask_items"
            elif "order_id" in missing:
                reply = "ask_order"
            else:
                reply = "report_error"
            tool = None
    # Loop breaker: the same template sent three times in a row means the conversation is stuck.
    recent = [m.get("text") for m in (obs.get("conversation") or [])[-6:] if m.get("role") == "assistant" and m.get("text")]
    if tool is None and reply not in (None, "transfer_notice", "closing") and len(recent) >= 3 and recent[-1] == recent[-2] == recent[-3] and "transfer_to_human_agents" in tools and not mem.get("transferred"):
        tool = "transfer_to_human_agents"
        args = {"summary": "Conversation is stuck; the agent could not identify the requested action from the user's messages. Last message: " + f["last_user_text"][:200]}
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
        elif reply == "ask_identity":
            text = "To get started, could you please provide the email address on your account, or your full name together with your zip code, so I can verify your identity?"
        elif reply == "ask_order":
            text = "Could you tell me the order id you are referring to (it starts with #W)? If you are not sure, I can look through your orders."
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
        elif reply == "llm" and allow_llm:
            llm_prompt = ("You are a retail customer-service agent. Follow this policy strictly: " + str(obs.get("policy"))[:6000] +
                          ". Known facts: " + str(f) + ". Conversation (latest last): " + str((obs.get("conversation") or [])[-10:]) +
                          ". Write the next message to the user only (no tool calls). Do not invent order details; ask for what you need.")
        else:
            text = "Could you tell me a bit more about what you would like to do with your order?"
        action = {"kind": "message", "text": text}
    return {"action": action, "memory": mem, "llm_prompt": llm_prompt, "llm_args_prompt": llm_args_prompt,
            "chosen": chosen, "confidence": confidence, "confirms": confirms}
'''

# ---------------------------------------------------------------------------------------------------------
# Node 3 (harness only): final. Merge generated text or arguments into the action.
# ---------------------------------------------------------------------------------------------------------
FINAL_SOURCE = '''
def run(obs, nodes, memory):
    d = nodes["decision"]
    action = dict(d["action"])
    mem = dict(d["memory"])
    draft = nodes.get("draft_reply")
    drafted_args = nodes.get("draft_args")
    if action.get("kind") == "message" and action.get("text") is None:
        text = draft.get("text") if isinstance(draft, dict) else None
        action["text"] = text or "Could you tell me a bit more about what you would like to do?"
    if action.get("kind") == "tool" and isinstance(drafted_args, dict):
        merged = dict(action.get("arguments") or {})
        for k, v in (drafted_args.get("arguments") or {}).items():
            if k not in merged or merged.get(k) in (None, ""):
                merged[k] = v
        action["arguments"] = merged
    return {"action": action, "memory": mem}
'''

REPLY_SCHEMA = {'type': 'object', 'required': ['text'], 'additionalProperties': False,
                'properties': {'text': {'type': 'string', 'minLength': 1, 'maxLength': 1500}}}
ARGS_SCHEMA = {'type': 'object', 'required': ['arguments'], 'additionalProperties': False,
               'properties': {'arguments': {'type': 'object'}}}


def _fill(source, allow_llm):
    return (source.replace('__ALLOW_LLM__', 'True' if allow_llm else 'False')
                  .replace('__WRITE_TOOLS__', json.dumps(list(WRITE_TOOLS)))
                  .replace('__TRANSFER_TEXT__', json.dumps(TRANSFER_TEXT)))


def _base_nodes(allow_llm):
    return [
        {'id': 'facts', 'kind': 'python', 'source': _fill(FACTS_SOURCE, allow_llm), 'inputs': []},
        {'id': 'judge', 'kind': 'jev', 'state': "nodes['facts']['state']", 'questions_expression': "nodes['facts']['questions']"},
        {'id': 'decision', 'kind': 'python', 'source': _fill(DECISION_SOURCE, allow_llm), 'inputs': ['facts', 'judge']},
    ]


def jev_only_seed():
    spec = {'format': FORMAT, 'name': 'tau2 retail · Jev-only seed', 'jev_model': JEV_MODEL,
            'nodes': _base_nodes(False),
            'output': "nodes['decision']['action']",
            'memory': "nodes['decision']['memory']"}
    return validate_spec(spec)


def harness_seed(llm_model='claude-sonnet-5'):
    nodes = _base_nodes(True) + [
        {'id': 'draft_reply', 'kind': 'llm', 'prompt': "nodes['decision']['llm_prompt']",
         'schema': copy.deepcopy(REPLY_SCHEMA), 'max_tokens': 600},
        {'id': 'draft_args', 'kind': 'llm', 'prompt': "nodes['decision']['llm_args_prompt']",
         'schema': copy.deepcopy(ARGS_SCHEMA), 'max_tokens': 400},
        {'id': 'final', 'kind': 'python', 'source': _fill(FINAL_SOURCE, True),
         'inputs': ['decision', 'draft_reply', 'draft_args']},
    ]
    spec = {'format': FORMAT, 'name': 'tau2 retail · harness seed', 'jev_model': JEV_MODEL, 'llm_model': llm_model,
            'nodes': nodes,
            'output': "nodes['final']['action']",
            'memory': "nodes['final']['memory']"}
    return validate_spec(spec)


SEEDS = {'jev_only': jev_only_seed, 'harness': harness_seed}
