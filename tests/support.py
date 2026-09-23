"""Scripted providers for tests. No network, deterministic, JevClient-shaped responses."""
import copy


class ScriptedJev:
    """Answer questions from a script: {question_id: {option: prob}} for choice/score, float for noul.

    Unscripted questions fall back to the first option / 0.5. Records every request.
    """
    mock = True
    transport = 'mock'

    def __init__(self, script=None):
        self.script = copy.deepcopy(script or {})
        self.requests = []
        self.stats = {'calls': 0, 'cache_hits': 0, 'input_tokens': 0, 'output_tokens': 0,
                      'elapsed_ms': 0.0, 'cost_usd': 0.0, 'unpriced_calls': 0}
        self.metadata = {'transport': 'mock', 'requested_model': 'typesafe-ai/jev',
                         'model_version_pinned': False, 'cache_namespace': 'scripted', 'cache_dir': None}

    def judge(self, state, questions, model=None):
        self.requests.append({'state': copy.deepcopy(state), 'questions': copy.deepcopy(questions)})
        self.stats['calls'] += 1
        answers = {}
        for qid, question in questions.items():
            scripted = self.script.get(qid)
            kind = question['type']
            if kind == 'noul':
                answers[qid] = {'type': 'noul', 'noul': float(scripted) if scripted is not None else 0.5}
            elif kind == 'choice':
                options = list(question['criteria'])
                probs = dict(scripted) if isinstance(scripted, dict) else {o: (1.0 if i == 0 else 0.0) for i, o in enumerate(options)}
                choice = max(probs, key=probs.get)
                answers[qid] = {'type': 'choice', 'choice': choice, 'probabilities': probs, 'confidence': probs[choice]}
            else:
                levels = len(question['criteria'])
                probs = dict(scripted) if isinstance(scripted, dict) else {str(i): (1.0 if i == 0 else 0.0) for i in range(levels)}
                score = sum(int(k) * v for k, v in probs.items())
                answers[qid] = {'type': 'score', 'score': score, 'probabilities': probs, 'confidence': max(probs.values())}
        return {'answers': answers, 'model': 'scripted', 'usage': {}, 'cache_hit': False,
                'elapsed_ms': 1.0, 'client_metadata': copy.deepcopy(self.metadata)}
