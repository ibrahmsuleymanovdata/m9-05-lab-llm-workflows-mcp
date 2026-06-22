"""
Lab | Build the Loop Yourself
Hand-rolled model -> tool -> model loop, with short-term memory
and a step limit. No agent framework — plain Python control flow.

Two tools (same as yesterday):
  - lookup_order(order_id): reads orders.json (private data the model can't know)
  - calculate(expression): exact arithmetic (the model is unreliable at this)
"""

import os
import json
import time
import google.generativeai as genai

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

with open("orders.json", "r") as f:
    ORDERS = json.load(f)

MAX_STEPS = 5  # the loop can never run more than this many times


# ---------------------------------------------------------------------------
# Step 1: the real functions your code will actually run
# ---------------------------------------------------------------------------

def lookup_order(order_id: str) -> dict:
    """Look up an order by ID in the local order database."""
    order = ORDERS.get(order_id)
    if order is None:
        # Don't crash. Return a clear, model-readable error instead.
        return {"error": f"Order '{order_id}' not found."}
    return order


def calculate(expression: str) -> dict:
    """Safely evaluate a simple arithmetic expression."""
    # Only allow digits, operators, parentheses, and whitespace —
    # this is the validation step: never blindly eval() model input.
    allowed_chars = set("0123456789.+-*/() ")
    if not set(expression) <= allowed_chars:
        return {"error": f"Expression '{expression}' contains disallowed characters."}
    try:
        result = eval(expression)  # safe here only because we whitelisted the charset above
        return {"result": result}
    except Exception as e:
        return {"error": f"Could not evaluate '{expression}': {e}"}


# ---------------------------------------------------------------------------
# Step 2: describe the tools to the model with a schema
# ---------------------------------------------------------------------------

lookup_order_schema = genai.protos.FunctionDeclaration(
    name="lookup_order",
    description="Look up an order by its order ID. Returns the item name, "
                "price, purchase date, and warranty length in months. "
                "Use this whenever the user asks about a specific order.",
    parameters=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "order_id": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="The order ID, e.g. 'A1001'.",
            ),
        },
        required=["order_id"],
    ),
)

calculate_schema = genai.protos.FunctionDeclaration(
    name="calculate",
    description="Evaluate a simple arithmetic expression (e.g. '1200 * 3') "
                "and return the exact numeric result. Use this for any "
                "math instead of computing it yourself.",
    parameters=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "expression": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="A simple arithmetic expression, e.g. '1200 * 3'.",
            ),
        },
        required=["expression"],
    ),
)

tools = genai.protos.Tool(function_declarations=[lookup_order_schema, calculate_schema])

AVAILABLE_FUNCTIONS = {
    "lookup_order": lookup_order,
    "calculate": calculate,
}

model = genai.GenerativeModel(model_name="models/gemini-3.5-flash", tools=[tools])


# ---------------------------------------------------------------------------
# Step 3: short-term memory — a plain list YOU control and resend every time
# ---------------------------------------------------------------------------

messages = []  # the entire running conversation lives here, in plain sight


# ---------------------------------------------------------------------------
# Step 4: the hand-rolled tool-use loop
# ---------------------------------------------------------------------------

def run_conversation(user_message: str):
    print(f"\n{'=' * 70}")
    print(f"USER: {user_message}")
    print(f"{'=' * 70}")

    # Append the new user turn to memory before doing anything else
    messages.append(
        genai.protos.Content(role="user", parts=[genai.protos.Part(text=user_message)])
    )

    for step in range(1, MAX_STEPS + 1):
        print(f"\n--- Step {step}/{MAX_STEPS} ---")

        # We call the model ourselves, sending the WHOLE messages list.
        # There is no hidden chat object doing this for us.
        response = model.generate_content(messages)

        # The model's reply also becomes part of memory, immediately.
        messages.append(response.candidates[0].content)

        function_calls = [
            part.function_call
            for part in response.candidates[0].content.parts
            if part.function_call
        ]

        if not function_calls:
            # No tool call -> model is done, this is the final answer.
            final_text = response.candidates[0].content.parts[0].text
            print(f"\nMODEL (final answer): {final_text}")
            return final_text

        # The model requested one or more tool calls — WE run them, not the model.
        function_responses = []
        for call in function_calls:
            name = call.name
            args = dict(call.args)
            print(f"  TOOL CALL -> {name}({args})")

            if name not in AVAILABLE_FUNCTIONS:
                result = {"error": f"Unknown tool '{name}'"}
            else:
                result = AVAILABLE_FUNCTIONS[name](**args)

            print(f"  TOOL RESULT <- {result}")

            function_responses.append(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=name,
                        response={"result": result},
                    )
                )
            )

        # Tool results go into memory too, then we loop again.
        messages.append(genai.protos.Content(role="user", parts=function_responses))
        time.sleep(3)

    # If we get here, we used up every step without a final answer.
    print(f"\nMODEL: couldn't finish in time (hit the {MAX_STEPS}-step limit).")
    return None


# ---------------------------------------------------------------------------
# Step 5: demo — two-turn conversation that PROVES memory works
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Turn 1: establishes the order in memory
    run_conversation("What did order A1001 cost?")

    time.sleep(20)  # avoid hitting per-minute rate limits

    # Turn 2: only answerable if the model still remembers turn 1 —
    # "them" and "it" refer back to A1001, which is never repeated here.
    run_conversation("And what about three of them?")

    # At this point `messages` holds the full transcript of both turns:
    # user -> model(tool_call) -> tool_result -> model(tool_call) -> tool_result
    # -> model(final) -> user -> model(final), all sent on every single call.
    print(f"\n{'=' * 70}")
    print(f"Final memory size: {len(messages)} entries in `messages`")
    print(f"{'=' * 70}")