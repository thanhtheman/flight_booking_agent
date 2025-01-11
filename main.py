from pydantic import BaseModel, Field
from datetime import datetime
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext



#creating structured answers that we expect the agent will produce
class FlightDetails(BaseModel):
    flight_number: str
    price: int
    origin: str = Field(description="Three-letter airport code")
    destination: str = Field(description="Three-letter airport code")
    date: datetime.date

class NonFlightFound(BaseModel):
    """when there is no flight found"""

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



