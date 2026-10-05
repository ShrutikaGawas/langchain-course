from typing import TYPE_CHECKING
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

if TYPE_CHECKING:
    from _typeshed import OpenBinaryMode

from langchain.chat_models import init_chat_model 
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


@tool
def get_product_price(product_name: str) -> str:
    """
    Lookup price of the product in the catalog and return the price as a string. If the product is not found, return 0.
    """
    print(f"Fetching price for product: {product_name}")
    prices = {
        "laptop": "$999",
        "smartphone": "$699",
        "headphones": "$199",
        "monitor": "$299"
    }
    return prices.get(product_name,0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount tier to the given price and return the discounted price as a string.
    """
    print(f">> Executing apply_discount with price: {price} and discount_tier: {discount_tier}")
    discount_percentages = {
        "bronze": 5,
        "silver": 12,
        "gold": 23
    }
    discount = discount_percentages.get(discount_tier, 0)
    print(f"Applying discount of {discount}% to price: {price}")
    return round(price * (1 - discount / 100), 2)

""" Agent Loop with LangChain Tool Calling Example """
@traceable(name="Langchain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question:{question}")
    print("="*60)

    messages = [
        SystemMessage(
            content=("You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=question)
    ]

    for iteration in range(MAX_ITERATIONS):
        print(f"Iteration {iteration }")

        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        # if no tool calls, this is the final answer
        if not tool_calls:
            print(f"Final Answer: {ai_message.content}")
            return ai_message.content

        # process only the first tool call for this iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        print(f"Tool Call: {tool_name} with args: {tool_args}")

        tool_use = tools_dict.get(tool_name)
        if tool_use is None:
            raise ValueError(f"Tool {tool_name} not found in tools_dict")
        observation = tool_use.invoke(tool_args)
        print(f"tool result: {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call_id
            )
        )
    print("ERROR: Max iterations reached without a final answer.")
    return None

if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")