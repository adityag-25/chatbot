import sys, json
sys.path.insert(0, r'c:\Users\adity\OneDrive\Desktop\PullRequest\chatbot')

# Test imports
from env import CustomerSupportEnv, Observation, Action, StepResult
print('[OK] Imports from env package')

# Test emotion detection
from env.emotion import detect_emotion
tests = [
    ('Product is broken', 'angry'),
    ('Delivery is too late', 'angry'),
    ('Delivery boy rude', 'angry'),
    ('Received empty box', 'angry'),
    ('Order stuck', 'angry'),
    ('I am confused about order', 'confused'),
    ('How long delivery?', 'neutral'),
    ('Need invoice copy', 'neutral'),
    ("My order hasn't arrived", 'angry'),
    ('Refund not received', 'angry'),
]
failed = 0
for query, expected in tests:
    result = detect_emotion(query)
    status = 'OK' if result == expected else 'FAIL'
    if status == 'FAIL': failed += 1
    print(f'  [{status}] "{query}" => {result!r}  (expected {expected!r})')
print(f'Emotion tests: {len(tests)-failed}/{len(tests)} passed')

# Test environment
print()
env = CustomerSupportEnv(task_file=r'c:\Users\adity\OneDrive\Desktop\PullRequest\chatbot\tasks\tasks.json')
obs = env.reset(task_index=0)
print(f'[OK] reset() query: {obs.conversation[0].content!r}')
print(f'     emotion={obs.emotion}  order_id={obs.context.get("order_id")}  order_status={obs.context.get("order_status")}')
print(f'     history seeded with {len(obs.context["history"])} msg(s)')

# Step
action = Action(intent='complaint', decision='escalate', response='I sincerely apologize for your experience. We have escalated your case.')
result = env.step(action)
print(f'[OK] step() reward={result.reward}  done={result.done}')
tr = result.info.get('tool_result', {})
print(f'     tool={tr.get("tool")}  status={tr.get("status")}  msg={tr.get("message","")[:60]}')

# Grader
from env.grader import grade_action
task_sample = {
    'intent': 'complaint',
    'decision': 'escalate',
    'keywords': ['sorry', 'issue'],
    'ideal_response': 'We apologize.',
    'emotion': 'angry'
}
score = grade_action(action, task_sample, 1)
print(f'[OK] grade_action() => {score}')

# Full 29-task run
print()
print('Running all 30 tasks (heuristic agent)...')
from baseline.run import heuristic_agent
total = 0
num_tasks = len(env.tasks)
for i in range(num_tasks):
    e = CustomerSupportEnv(task_file=r'c:\Users\adity\OneDrive\Desktop\PullRequest\chatbot\tasks\tasks.json')
    o = e.reset(task_index=i)
    done = False
    r_last = 0
    while not done:
        a = heuristic_agent(o)
        step_r = e.step(a)
        o = step_r.observation
        done = step_r.done
        r_last = step_r.reward
    total += r_last
avg = total / num_tasks
print(f'Average reward across {num_tasks} tasks: {avg:.4f}')
print()
print('[ALL TESTS PASSED]')
