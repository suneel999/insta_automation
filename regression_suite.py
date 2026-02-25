"""
Lightweight regression suite for Instagram DM assistant.
Runs multi-turn scenarios and validates expected keywords in replies.
"""

from ai_handler import (
    get_on_ai_response,
    USER_STATES,
    CONVERSATION_HISTORY,
)


def reset_state():
    USER_STATES.clear()
    CONVERSATION_HISTORY.clear()


def run_scenario(scenario_id, turns):
    uid = f"reg_{scenario_id}"
    for idx, turn in enumerate(turns, start=1):
        user_text = turn["user"]
        expected_any = [e.lower() for e in turn.get("expect_any", [])]
        forbidden = [f.lower() for f in turn.get("forbid", [])]

        bot = get_on_ai_response(user_text, uid).lower()

        if expected_any and not any(token in bot for token in expected_any):
            return False, idx, user_text, bot, f"Missing expected token(s): {expected_any}"
        if forbidden and any(token in bot for token in forbidden):
            return False, idx, user_text, bot, f"Contains forbidden token(s): {forbidden}"
    return True, None, None, None, None


SCENARIOS = [
    [
        {"user": "hi", "expect_any": ["welcome", "hello"]},
        {"user": "price", "expect_any": ["which product", "product price"]},
        {"user": "whey", "expect_any": ["2lb", "5lb"]},
        {"user": "2lb", "expect_any": ["which country", "uae, ksa, or egypt"]},
        {"user": "uae", "expect_any": ["aed 156.45"]},
        {"user": "and in ksa", "expect_any": ["sar 248.40"]},
    ],
    [
        {"user": "prce", "expect_any": ["which product", "product price"]},
        {"user": "hidro", "expect_any": ["which country", "uae, ksa, or egypt"]},
        {"user": "egpyt", "expect_any": ["1750 le"]},
        {"user": "and saoodi", "expect_any": ["sar 441.60"]},
    ],
    [
        {"user": "show me protein", "expect_any": ["gold standard 100% whey", "platinum hydrowhey"]},
        {"user": "gold standrad cost", "expect_any": ["2lb", "5lb"]},
        {"user": "5lb", "expect_any": ["which country", "uae, ksa, or egypt"]},
        {"user": "ksa", "expect_any": ["sar 389.85"]},
    ],
    [
        {"user": "price isolate", "expect_any": ["which country", "uae, ksa, or egypt"]},
        {"user": "ksa", "expect_any": ["sar 362.25"]},
        {"user": "what about egypt", "expect_any": ["2,200 le"]},
    ],
    [
        {"user": "authntic?", "expect_any": ["sticker", "originalon.com"]},
        {"user": "veagn?", "expect_any": ["vegan protein", "do not currently offer vegan"]},
    ],
    [
        {"user": "price pre workout", "expect_any": ["which country", "uae, ksa, or egypt"]},
        {"user": "egypt", "expect_any": ["650 le"]},
        {"user": "and ksa", "expect_any": ["do not have a ksa price", "check www.sporter.com"]},
    ],
    [
        {"user": "what products are you having", "expect_any": ["main categories"]},
        {"user": "vitmins", "expect_any": ["opti-men", "fish oil softgels"]},
        {"user": "fish oil price", "expect_any": ["which country", "uae, ksa, or egypt"]},
        {"user": "uae", "expect_any": ["aed 72.45"]},
    ],
    [
        {"user": "price serious mass", "expect_any": ["which country", "uae, ksa, or egypt"]},
        {"user": "egypt", "expect_any": ["1650 le"]},
        {"user": "and in uae", "expect_any": ["aed 157.50"]},
    ],
    [
        {"user": "how much glutamine in ksa", "expect_any": ["sar 205.85"]},
    ],
    [
        {"user": "gold standard whey 5lb ksa price", "expect_any": ["sar 389.85"]},
    ],
]


def main():
    reset_state()
    passed = 0
    failed = 0

    for i, scenario in enumerate(SCENARIOS, start=1):
        ok, turn_idx, user_text, bot_text, reason = run_scenario(i, scenario)
        if ok:
            passed += 1
        else:
            failed += 1
            print(f"[FAIL] Scenario {i}, turn {turn_idx}")
            print(f"  user: {user_text}")
            print(f"  bot : {bot_text}")
            print(f"  err : {reason}")

    total = passed + failed
    rate = (passed / total) * 100 if total else 0
    print(f"\nPass: {passed}/{total} ({rate:.1f}%)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

