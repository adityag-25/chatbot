from env.environment import CustomerSupportEnv
from env.models import Action

def run_simulation():
    env = CustomerSupportEnv()
    total_reward = 0
    for task_num in range(5):   # runs 5 tasks
        print("\nRunning Task:", task_num)

        obs = env.reset()
        done = False
        turn = 0
        while not done:
            history = obs.context["history"]
            text = str(history).lower()

            if "refund" in text:
                decision = "refund"
                intent = "refund_request"
            elif "delay" in text or "angry" in text:
                decision = "escalate"
                intent = "complaint"
            else:
                decision = "resolve"
                intent = "info_request"

            response_text = "Sorry for the inconvenience. Let me help you."

            action = Action(
                intent=intent,
                decision=decision,
                response=response_text
            )

            result = env.step(action)

            obs = result.observation
            done = result.done

            print("Reward:", result.reward)

            turn += 1
        total_reward += result.reward
    print("\nAverage score:", total_reward / 5,"\n")


if __name__ == "__main__":
    print("Running baseline simulation...\n")
    run_simulation()