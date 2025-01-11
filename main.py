from pydantic import BaseModel, Field
import datetime
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext, ModelRetry
from dotenv import load_dotenv
from typing_extensions import Literal
from pydantic_ai.usage import Usage, UsageLimits
from pydantic_ai.messages import ModelMessage
from rich.prompt import Prompt
import asyncio

load_dotenv()

#tracking usage
def print_usage_stats(usage: Usage, label: str = "Current"):
    print(f"\n{label} Usage Stats:")
    print(f"Requests: {usage.requests}")
    print(f"Request Tokens: {usage.request_tokens}")
    print(f"Response Tokens: {usage.response_tokens}")
    print(f"Total Tokens: {usage.total_tokens}")
    if usage.details:
        print("Additional Details:", usage.details)

#creating structured answers that we expect the agent will produce
class FlightDetails(BaseModel):
    flight_number: str
    price: int
    origin: str = Field(description="Three-letter airport code")
    destination: str = Field(description="Three-letter airport code")
    date: datetime.date

class NonFlightFound(BaseModel):
    """when there is no flight found"""

#createing structured answers for seat selection
class SeatPreference(BaseModel):
    row: int = Field(ge=1, le=30)
    seat: Literal['A', 'B', 'C', 'D', 'E', 'F']
class Failed(BaseModel):
    """unable to extract a seat selection"""

#dependencies that the agent need
@dataclass
class Deps:
    web_page_text: str
    req_origin: str
    req_destination: str
    req_date: datetime.date

search_agent = Agent("openai:gpt-4o", 
                     deps_type=Deps, 
                     result_type=FlightDetails | NonFlightFound, 
                     retries=4,
                     system_prompt="Your job is to find the cheapest flight for the user on the given date.")

extraction_agent = Agent("openai:gpt-4o", 
                         result_type=list[FlightDetails], 
                         system_prompt="Extract all the flight details from the given text.")


@search_agent.tool
async def extract_flights(ctx: RunContext[Deps]) -> list[FlightDetails]:
    result = await extraction_agent.run(ctx.deps.web_page_text, usage=ctx.usage)
    return result.data

#checking if the result created by the agent meet the user's requirements
@search_agent.result_validator
async def validate_result(ctx: RunContext[Deps], result: FlightDetails | NonFlightFound) -> FlightDetails | NonFlightFound:
    if isinstance(result, NonFlightFound):
        return result
    errors: list[str] = []
    if result.origin != ctx.deps.req_origin:
        errors.append(f"Flight should have origin {ctx.deps.req_origin}, not {result.origin}")
    if result.destination != ctx.deps.req_destination:
        errors.append(f"Flight should have destination {ctx.deps.req_destination} not {result.destination}")
    if result.date != ctx.deps.req_date:
        errors.append(f"Flight should have date {ctx.deps.date} not {result.date}")
    if errors:
        raise ModelRetry("\n".join(errors))
    else:
        return result


# in reality this would be downloaded data from a booking site (maybe by another agent)
flights_web_page = """
1. Flight SFO-AK123
- Price: $350
- Origin: San Francisco International Airport (SFO)
- Destination: Ted Stevens Anchorage International Airport (ANC)
- Date: January 10, 2025

2. Flight SFO-AK456
- Price: $370
- Origin: San Francisco International Airport (SFO)
- Destination: Fairbanks International Airport (FAI)
- Date: January 10, 2025

3. Flight SFO-AK789
- Price: $400
- Origin: San Francisco International Airport (SFO)
- Destination: Juneau International Airport (JNU)
- Date: January 20, 2025

4. Flight NYC-LA101
- Price: $250
- Origin: San Francisco International Airport (SFO)
- Destination: Ted Stevens Anchorage International Airport (ANC)
- Date: January 10, 2025

5. Flight CHI-MIA202
- Price: $200
- Origin: Chicago O'Hare International Airport (ORD)
- Destination: Toronto Pearson International Airport(YYZ)
- Date: January 12, 2025

6. Flight BOS-YYZ303
- Price: $120
- Origin: Boston Logan International Airport (BOS)
- Destination: Toronto Pearson International Airport(YYZ)
- Date: January 10, 2025

7. Flight BOS-YYZ404
- Price: $250
- Origin: Boston Logan International Airport (BOS)
- Destination: Toronto Pearson International Airport(YYZ)
- Date: January 10, 2025

8. Flight ATL-HOU505
- Price: $180
- Origin: Hartsfield-Jackson Atlanta International Airport (ATL)
- Destination: George Bush Intercontinental Airport (IAH)
- Date: January 10, 2025
"""

# restrict how many requests this app can make to the LLM
usage_limits = UsageLimits(request_limit=15)

seat_preference_agent = Agent("openai:gpt-4o", result_type=SeatPreference | Failed,
                              system_prompt=(
                                "Extract the user's seat preference. "
                                'Seats A and F are window seats. '
                                'Row 1 is the front row and has extra leg room. '
                                'Rows 14, and 20 also have extra leg room. '))

async def main():
    deps = Deps(
        web_page_text=flights_web_page,
        req_date=datetime.date(2025,1,10),
        req_origin='BOS',
        req_destination='YYZ'
    )
    message_history: list[ModelMessage] | None = None
    usage: Usage = Usage()
    print_usage_stats(usage, "Initial")
    counter = 0
    while True:
        # we ask the LLM to do a direct search by giving it a simple json list of flights and there is no surprise that it makes mistake
        result = await search_agent.run(
            f"find me a flight  from {deps.req_origin} to {deps.req_destination}",
            deps=deps,
            usage=usage,
            message_history=message_history)
        counter+= 1
        print(f"Agent run {counter} time")
        if isinstance(result.data, NonFlightFound):
            print("No Flight found!")
            print_usage_stats(usage, f"Updated {counter}")
            break
        else:
            flight = result.data
            print(f"I found you this flight: \n {flight}\n")
            answer = Prompt.ask("Do you want to buy this flight or keep searching? please type 'buy' or 'search'",
                                choices=['buy', 'search', ''],
                                show_choices=False)
            if answer == "buy":
                seat = await find_seat(usage)
                await buy_ticket(flight, seat)
                print_usage_stats(usage, f"Updated {counter}")
                break
            else:
                message_history = result.all_messages(
                    result_tool_return_content='Please suggest another flight'
                )
            print_usage_stats(usage, f"Updated {counter}")


async def find_seat(usage: Usage) -> SeatPreference:
    message_history: list[ModelMessage] | None = None
    while True:
        answer = Prompt.ask("What seat do you want?")
        result = await seat_preference_agent.run(
                                                user_prompt=answer,
                                                message_history=message_history,
                                                usage=usage,
                                                usage_limits=usage_limits
                                                )
        if isinstance(result.data, SeatPreference):
            return result.data
        else:
            print("Could not understand seat preference ")
            message_history = result.all_messages_json()

async def buy_ticket(flight_details: FlightDetails, seat: SeatPreference) -> str:
    print(f"Purchasing Flight: \n {flight_details} \n Seat: {seat}\n")




asyncio.run(main())

