from pydantic import BaseModel
import random
import string
import dspy
import os
from api_keys import TOGETHER_API_KEY, SERPAPI_API_KEY
import util
import numpy as np
from synthetic_users import SYNTHETIC_USERS
from typing import Optional
from serpapi import GoogleSearch
from bs4 import BeautifulSoup
from mem0 import Memory
from datetime import datetime
import time

def greeting(self):
    """Return a message that the chatbot uses to greet the user."""
    ########################################################################
    # TODO: Write a short greeting message                                 #
    ########################################################################

    greeting_message = "Hi, I'm your movie ticket assistant! I can recommend movies, check showtimes, and help you book tickets. How can I help you today?"

    ########################################################################
    #                             END OF YOUR CODE                         #
    ########################################################################
    return greeting_message

os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY
os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY

## load ratings matrix and convert user ratings to binary in user_ratings_dict
# note that the users in ratings_matrix are different from the users in user_ratings_dict
titles, ratings_matrix = util.load_ratings('data/ratings.txt')
user_ratings_dict = {user: np.zeros(len(titles)) for user in SYNTHETIC_USERS}
for user, movies in SYNTHETIC_USERS.items():
    for movie in movies:
        user_ratings_dict[user][titles.index(movie)] = 1

class Date(BaseModel):
    # Somehow LLM is bad at specifying `datetime.datetime`, so
    # we define a custom class to represent the date.
    year: int
    month: int
    day: int
    hour: int
    minute: int

class UserProfile(BaseModel):
    name: str
    email: str
    balance: float

class Movie(BaseModel):
    title: str
    start_time: Date
    price: float

class Ticket(BaseModel):
    user_name: str
    movie_title: str
    time: Date

class Request(BaseModel):
    user_request: str
    user_name: str

user_database = {
    "peter": UserProfile(name="Peter", email="peter@gmail.com", balance=42),
    "emma": UserProfile(name="Emma", email="emma@gmail.com", balance=79),
    "jake": UserProfile(name="Jake", email="jake@gmail.com", balance=13),
    "sarah": UserProfile(name="Sarah", email="sarah@gmail.com", balance=36),
    "michael": UserProfile(name="Michael", email="michael@gmail.com", balance=8),
    "lisa": UserProfile(name="Lisa", email="lisa@gmail.com", balance=97),
    "marcus": UserProfile(name="Marcus", email="marcus@gmail.com", balance=59),
    "sophia": UserProfile(name="Sophia", email="sophia@gmail.com", balance=25),
    "chris": UserProfile(name="Chris", email="chris@gmail.com", balance=63),
    "amy": UserProfile(name="Amy", email="amy@gmail.com", balance=91),
}

showtime_database = {
    "Back to the Future": Movie(title="Back to the Future (1985)", start_time=Date(year=2025, month=11, day=13, hour=10, minute=0), price=15.0),
    "Speed": Movie(title="Speed (1994)", start_time=Date(year=2025, month=11, day=13, hour=11, minute=30), price=20.0),
    "Star Wars: Episode VI - Return of the Jedi": Movie(title="Star Wars: Episode VI - Return of the Jedi (1983)", start_time=Date(year=2025, month=11, day=15, hour=13, minute=0), price=18.0),
    "Terminator": Movie(title="Terminator, The (1984)", start_time=Date(year=2025, month=11, day=15, hour=18, minute=0), price=14.0),
    "Star Wars: Episode V - The Empire Strikes Back": Movie(title="Star Wars: Episode V - The Empire Strikes Back (1980)", start_time=Date(year=2025, month=11, day=15, hour=20, minute=0), price=16.5),
    "Matrix": Movie(title="Matrix, The (1999)", start_time=Date(year=2025, month=11, day=15, hour=22, minute=0), price=19.0),
    "Silence of the Lambs": Movie(title="Silence of the Lambs, The (1991)", start_time=Date(year=2025, month=11, day=16, hour=10, minute=15), price=17.0),
    "Fight Club": Movie(title="Fight Club (1999)", start_time=Date(year=2025, month=11, day=16, hour=12, minute=45), price=18.5),
    "Lord of the Rings: The Two Towers": Movie(title="Lord of the Rings: The Two Towers, The (2002)", start_time=Date(year=2025, month=11, day=16, hour=15, minute=0), price=17.5),
    "Lord of the Rings: The Fellowship of the Ring": Movie(title="Lord of the Rings: The Fellowship of the Ring, The (2001)", start_time=Date(year=2025, month=11, day=16, hour=17, minute=30), price=17.0),
    "Pulp Fiction": Movie(title="Pulp Fiction (1994)", start_time=Date(year=2025, month=11, day=16, hour=19, minute=45), price=15.5),
    "Star Wars: Episode IV - A New Hope": Movie(title="Star Wars: Episode IV - A New Hope (1977)", start_time=Date(year=2025, month=11, day=16, hour=22, minute=0), price=16.0),
    "Titanic": Movie(title="Titanic (1997)", start_time=Date(year=2025, month=11, day=15, hour=10, minute=0), price=20.0)
}

ticket_database = {}
request_database = {}


################################################################################################################################################
# PART 1

## defining tools and helper functions for the tools

def _generate_id(length=8):
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=length))

def similarity(u, v):
    """
    Calculate the cosine similarity between two vectors.
    You may assume that the two arguments have the same shape.
    :param u: one vector, as a 1D numpy array
    :param v: another vector, as a 1D numpy array
    :returns: the cosine similarity between the two vectors
    Note: you should return 0 if u or v has norm 0
    """
    ########################################################################
    # TODO: Compute cosine similarity between the two vectors.             #
    ########################################################################
    similarity = 0
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)

    if u_norm == 0 or v_norm == 0:
        similarity = 0
    else:
        similarity = np.dot(u, v) / (u_norm * v_norm)
    ########################################################################
    #                          END OF YOUR CODE                            #
    ########################################################################
    return similarity

def recommend_movies(user_name: str, k=3):
    """
    Generate a list of indices of movies to recommend using collaborative
        filtering.

    You should return a collection of `k` indices of movies recommendations.

    As a precondition, user_ratings have been loaded for you based on the provided user_name.

    Do not recommend movies that the user has already rated (since the ratings of the rated movies are used to calculate similarity to a potential recommendation)
    If the user already rated every movie, then the function should return an empty list.

    Hint: the similarity between two movies is based on the similarity of their ratings in ratings_matrix

    :returns: a list of k movie titles corresponding to movies in
    ratings_matrix, in descending order of recommendation. (the k movie titles correspond to "movie indices" in ratings_matrix)
    """
    user_profile = user_database[user_name.lower()]
    user_name = user_profile.name
    user_ratings = user_ratings_dict[user_name]

    ########################################################################
    # TODO: Implement collaborative filtering to generate a list of movie  #
    # indices to recommend to the user.                                    #
    ########################################################################
    # Populate this list with k movie indices to recommend to the user.    
    recommendations = []
    user_ratings_arr = np.atleast_1d(user_ratings)

    rated_movies = np.where(user_ratings_arr == 1)[0]
    unrated_movies = np.where(user_ratings_arr == 0)[0]

    if len(unrated_movies) > 0:
        scores = []

        for movie_idx in unrated_movies:
            score = 0
            for rated_idx in rated_movies:
                score += similarity(ratings_matrix[movie_idx], ratings_matrix[rated_idx])
            scores.append((movie_idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        recommendations = [movie_idx for movie_idx, _ in scores[:k]]
    
    ########################################################################
    #                          END OF YOUR CODE                            #
    ########################################################################

    ## convert the movie indices to movie titles
    result_titles = [titles[movie_index] for movie_index in recommendations]
    return result_titles

def general_qa(user_request: str):
    """
    Answer a general question about movies by making an LLM call.
    """
    lm = dspy.LM("together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1")
    dspy.configure(lm=lm)

    response = lm(messages=[{"role": "user", "content": user_request}])
    return response

def find_time(movie_title: str):
    """
    Find the time of the given movie title. Title must be one of: 
    [Back to the Future, Speed, Star Wars: Episode VI - Return of the Jedi, 
    Terminator, Star Wars: Episode V - The Empire Strikes Back, Matrix, 
    Silence of the Lambs, Fight Club, Lord of the Rings: The Two Towers, 
    Lord of the Rings: The Fellowship of the Ring, Pulp Fiction, 
    Star Wars: Episode IV - A New Hope, Titanic]
    """
    movie = showtime_database[movie_title]
    return movie.start_time

def find_price(movie_title: str):
    """
    Find the price of the given movie title. Title must be one of: 
    [Back to the Future, Speed, Star Wars: Episode VI - Return of the Jedi, 
    Terminator, Star Wars: Episode V - The Empire Strikes Back, Matrix, 
    Silence of the Lambs, Fight Club, Lord of the Rings: The Two Towers, 
    Lord of the Rings: The Fellowship of the Ring, Pulp Fiction, 
    Star Wars: Episode IV - A New Hope, Titanic]
    """
    movie = showtime_database[movie_title]
    return movie.price

def find_balance(user_name: str):
    """
    Find the balance of the given user name. 
    Name must be one of: [peter, emma, jake, sarah, michael, lisa, marcus, sophia, chris, amy]
    """
    user_profile = user_database[user_name.lower()]
    return user_profile.balance

def file_request(user_request: str, user_name: str):
    """
    File a human customer support request if this is something the agent cannot handle.
    """
    request_id = _generate_id(length=6)
    request_database[request_id] = Request(
        user_request=user_request,
        user_name=user_name,
    )
    return request_id


def book_ticket(user_name: str, movie_title: str):
    """
    Book a ticket for the given user and movie title. Tile must be one of: 
    [Back to the Future, Speed, Star Wars: Episode VI - Return of the Jedi, 
    Terminator, Star Wars: Episode V - The Empire Strikes Back, Matrix, 
    Silence of the Lambs, Fight Club, Lord of the Rings: The Two Towers, 
    Lord of the Rings: The Fellowship of the Ring, Pulp Fiction, 
    Star Wars: Episode IV - A New Hope, Titanic]
    """
   
    ########################################################################
    # TODO: Implement the `book_ticket` tool                                
    # * Only make a booking if the user has enough balance. Then, update the 
    #   user's balance in the user's profile and add new ticket booking to the 'ticket_database'.
    #  If there is not enough balance, return: "Insufficient balance to book the ticket for {movie_title}."
    # * Use `_generate_id` to create a 6-digit ticket number for the booking
    # * For any requests that can't be handled by your agent, make a human 
    #   customer support request by calling the `file_request` tool 
    #   to add the request to the `request_database`
    ########################################################################
    ticket_number = '0'
    user_balance = None

    try:
        user_profile = user_database[user_name.lower()]
        movie = showtime_database[movie_title]
    except:
        file_request(f"Book ticket for {movie_title}", user_name)
        return f"Unable to handle booking request for {movie_title}. A customer support request has been filed."

    if user_profile.balance < movie.price:
        return f"Insufficient balance to book the ticket for {movie_title}."

    ticket_number = _generate_id(length=6)
    user_profile.balance -= movie.price
    user_balance = user_profile.balance

    ticket_database[ticket_number] = Ticket(
        ticket_number=ticket_number,
        user_name=user_profile.name,
        movie_title=movie_title,
        time=movie.start_time,
        price=movie.price,
    )
    ########################################################################
    #                          END OF YOUR CODE                            #
    ########################################################################
    return f"Ticket booked successfully for {user_name} for the movie {movie_title}. The ticket number is {ticket_number}. Your new balance is {user_balance}."


## Integrating tools into an LLM agent: you will use the agent below for part 1

# The MovieTicketAgent class is a wrapper that modifies dspy.Signature. If you are curious
# about the signature, read the documentation here:
# https://dspy.ai/learn/programming/signatures/#class-based-dspy-signatures

class MovieTicketAgent(dspy.Signature):
    ########################################################################
    ## TODO: Add a few sentences to flesh out the agent objective in the docstring below.
    # In DSPy, the docstring of a Signature acts as the system prompt 
    # for the language model. It defines the agent’s role, constraints, 
    # and decision-making strategy. So it is crucial to define it well!
    # Hint: you can add details about what tools the agent will need to call 
    # in order to successfully complete the tasks
    ########################################################################
    """
    You are a movie ticket agent that helps users recommend movies, answer general movie questions, check showtimes, check ticket prices, check account balances, and book movie tickets.
    
    Use the provided tools to answer the user's request. Choose only from the tool arguments that actually exist in each tool signature. Do not invent extra tool arguments.
    
    If the user asks for movie recommendations, use recommend_movies with user_name and k only, then finish after returning the recommendations. Do not call general_qa to ask follow-up questions after recommendations are returned.
    
    Use general_qa only for open-ended movie questions such as plot summaries or movie information. Use find_time for showtimes, find_price for prices, find_balance for account balances, and book_ticket to book tickets.
    
    Do not simulate a user response or continue the interaction on your own. Once the user's request has been fulfilled, finish and provide a clear final answer.
    """
    ########################################################################
    #                          END OF YOUR CODE                            #
    ########################################################################
    
    user_request: str = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc="Message that summarizes the process result, and the information users need, e.g., the ticket number if a new ticket is booked."
    )

dspy.configure(lm=dspy.LM("together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1"))

react_agent = dspy.ReAct(
    MovieTicketAgent,
    tools = [
        recommend_movies,
        general_qa,
        ########################################################################
        ## TODO: add other tools for your agent here
        ########################################################################
        find_time,
        find_price,
        find_balance,
        book_ticket,
        file_request,
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
    ]
)


################################################################################################################################################
# PART 2

## Part 2: web search utilities 

def extract_text(html: str) -> str:
    """
    Extracts clean, readable text from raw HTML.

    This function takes an HTML page (returned from a web request) and removes
    non-readable content. 

    Args:
        html (str): Raw HTML content of a web page.

    Returns:
        str: Cleaned plain-text version of the page content.
    """
    soup = BeautifulSoup(html, "html.parser")   # Parse the raw HTML into a structured BeautifulSoup object
    for tag in soup(["script", "style", "noscript"]): # Remove tags that do not contain meaningful visible text
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)  # Extract all remaining visible text from the HTML
    return " ".join(text.split())


class WebTools:
    """
    Utility class that provides web search and page-reading tools for the agent.

    This class acts as a thin wrapper around a web search API (SerpAPI).
    The agent calls these methods when it needs *external information*
    that is not available in its prompt or memory.

    Conceptually:
    - The LLM decides *when* to search
    - This class handles *how* the search is executed
    - The returned text is formatted to be readable by both humans and LLMs
    """

    def __init__(self, serpapi_key: Optional[str] = None):
        self.serpapi_key = serpapi_key or os.getenv("SERPAPI_API_KEY")

    def web_search(self, query: str, num_results: int = 5, page: int = 1) -> str:
        """
        Search the web for current movie-related information.

        Use this tool for release dates, cast, directors, sequels, and other
        up-to-date movie facts.

        Args:
          query: search query string
          num_results: number of results to return (recommended <= 10)
          page: pagination index starting from 1
        Returns:
          A formatted list of results with title, link, and snippet.
        """
        if not self.serpapi_key:
            return "Error: SERPAPI_API_KEY is not set."

        # Bing via SerpAPI. Pagination is controlled by 'first' for bing.
        # page=1 => first=0, page=2 => first=num_results, etc.
        first = (max(page, 1) - 1) * num_results
        # Parameters passed to the SerpAPI search endpoint.
        # We use Bing here, but SerpAPI supports multiple search engines.
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.serpapi_key,
            "count": num_results,
            "first": first,
        }

        try:
            results = GoogleSearch(params).get_dict()  # Execute the search request and parse the JSON response
            organic = results.get("organic_results", []) or []
            if not organic:
                return "No results found."
            # Build a human- and LLM-readable summary of the results
            lines = [f"Web search results for: {query} (page {page})"]
            for i, item in enumerate(organic[:num_results], 1):
                title = item.get("title") or "(no title)"
                link = item.get("link") or "(no link)"
                snippet = item.get("snippet") or ""
                
                page_text = ""
            if link != "(no link)":
                try:
                    response = requests.get(link, timeout=5, headers={
                        "User-Agent": "Mozilla/5.0"
                    })
                    if response.ok:
                        page_text = extract_text(response.text)[:1000]
                except Exception:
                    page_text = ""

            lines.append(f"{i}. {title}")
            lines.append(f"   Link: {link}")
            if snippet:
                lines.append(f"   Snippet: {snippet}")
            if page_text:
                lines.append(f"   Page text: {page_text}")

            return "\n".join(lines)

        except Exception as e:
            return f"Error during web_search: {str(e)}"



# memory functionalities
## Memory configuration
memory_config = {
    "llm": {
        "provider": "together",
        "config": {
            "model": "Qwen/Qwen3-Next-80B-A3B-Instruct",
            "temperature": 0.1
        }
    },
    "embedder": {
        "provider": "together",
        "config": {
            "model": "intfloat/multilingual-e5-large-instruct"
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "embedding_model_dims": 1024
        }
    }
}


## Part 2: memory utilities
class MemoryTools:
    """Tools for interacting with the Mem0 memory system."""

    DEFAULT_USER_ID = "default_user"

    def __init__(self, memory: Memory):
        self.memory = memory

    def store_memory(self, content: str, user_id: str = "default_user") -> str:
        """
        Store a piece of information in memory.

        This is typically called when the agent learns something that should
        persist across turns (e.g., user preferences, reminders, personal facts).

        Args:
            content (str): The text to store in memory.
            user_id (str): Identifier for the user whose memory this belongs to.

        Returns:
            str: A confirmation message or an error message.
        """
        try:
            ########################################################################
            # TODO: add the content to Mem0's memory store for user_id
            # Hint: It may be helpful to review mem0's memory operations here:
            # https://docs.mem0.ai/core-concepts/memory-operations
            ########################################################################
            self.memory.add(
                [{"role":"user", "content": content}],
                user_id=user_id
            )
            ########################################################################
            #                          END OF YOUR CODE                            #
            ########################################################################
            return f"Stored memory: {content}"
        except Exception as e:
            return f"Error storing memory: {str(e)}"

    def create_memory(self, results):
        """
        Helper function that creates a memory_text string containing all of the memories. You will call this function
        in search_memories and get_all_memories().
        This function should return memory_text
        """
        memory_text = "Relevant memories found:\n"
        for i, result in enumerate(results["results"]):
            memory_text += f"{i}. {result['memory']}\n"
        return memory_text

    def search_memories(self, query: str, user_id: str = "default_user", limit: int = 5) -> str:
        """
        Search memory for items relevant to a query.

        This is used when the agent needs to recall previously stored information,
        such as user preferences or earlier statements.

        Use this when the user asks:
        - what is my favorite ...
        - what did I tell you before ...
        - when did I ...

        Args:
            query (str): Natural-language search query.
            user_id (str): Identifier for the user whose memory should be searched.
            limit (int): Maximum number of memories to return.

        Returns:
            str: A formatted list of relevant memories or a message indicating
            that nothing was found.
        """
        try:
            ########################################################################
            # TODO: search for relevant memories and store them in results
            # Hint: it would be helpful to read the documentation of 
            # mem0 to see how to use the `search` method: https://github.com/mem0ai/mem0
            ########################################################################
            results = self.memory.search(query=query, user_id=user_id, limit=limit)
            ########################################################################
            #                          END OF YOUR CODE                            #
            ########################################################################
            if not results or "results" not in results or len(results["results"]) == 0:
                return "No relevant memories found."
            memory_text = self.create_memory(results)
            return memory_text
        except Exception as e:
            return f"Error searching memories: {str(e)}"

    def get_all_memories(self, user_id: str = "default_user") -> str:
        """
        Get all memories for a user.
        Use this when the user asks what you remember about them.
        """
        try:
            results = self.memory.get_all(user_id=user_id)
            if not results or "results" not in results or len(results["results"]) == 0:
                return "No memories found for this user."

            memory_text = self.create_memory(results)
            return memory_text
        except Exception as e:
            return f"Error retrieving memories: {str(e)}"
    
    def update_memory(self, memory_id: str, new_content: str) -> str:
        """Update an existing memory."""
        try:
            ########################################################################
            # TODO: Replace the old memory content with the new content
            # Hint: It may be helpful to review mem0's memory operations here:
            # https://docs.mem0.ai/core-concepts/memory-operations
            ########################################################################
            self.memory.update(memory_id=memory_id, data=new_content)
            ########################################################################
            #                          END OF YOUR CODE                            #
            ########################################################################
            return f"Updated memory with new content: {new_content}"
        except Exception as e:
            return f"Error updating memory: {str(e)}"

    def delete_memory(self, memory_id: str) -> str:
        """Delete a specific memory."""
        try:
            ########################################################################
            # TODO: delete the memory for a given memory_id
            # Hint: It may be helpful to review mem0's memory operations here:
            # https://docs.mem0.ai/core-concepts/memory-operations
            ########################################################################
            self.memory.delete(memory_id=memory_id)
            ########################################################################
            #                          END OF YOUR CODE                            #
            ########################################################################
            return "Memory deleted successfully."
        except Exception as e:
            return f"Error deleting memory: {str(e)}"


# other helper functions
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def set_reminder(reminder_text: str, date_time: str = None, user_id: str = "default_user") -> str:
    """Set a reminder for the user."""
    reminder = f"Reminder set for {date_time}: {reminder_text}"
    # This will be connected to memory_tools in the agent
    return reminder


def get_preferences(category: str = "general", user_id: str = "default_user") -> str:
    """Get user preferences for a specific category."""
    # This will be connected to memory_tools in the agent
    return f"Getting preferences for {category}"


def update_preferences(category: str, preference: str, user_id: str = "default_user") -> str:
    """Update user preferences."""
    # This will be connected to memory_tools in the agent
    return f"Updated {category} preference to {preference}"


## You will use the enhanced agent below for part 2
class EnhancedMovieTicketAgent(dspy.Module):
    """Movie ticket agent with web search and memory capabilities.  You have access to web search to find current movie information, memory to remember user preferences,
    and various tools to handle user requests. You should decide the right tool to use in order to
    fulfill users' request.
    
    Use memory tools whenever the user shares personal preferences, facts, or
    past experiences that may be useful later.

    Rules:
    - If the user says 'remember', 'my favorite', 'I like', 'I prefer', or shares
      a personal fact or preference, use store_memory.
    - If the user asks 'what is my favorite ...', 'what do you remember about me',
      or asks about something they told you before, use search_memories or
      get_all_memories.
    - Use web_search for current or up-to-date movie information.
    - Do not use general_qa when the answer should come from memory.
    """

    def __init__(self, enable_web_search=True, enable_memory=True):
        super().__init__()
        
        # Initialize web tools
        self.web_tools = WebTools() if enable_web_search else None
        
        # Initialize memory
        if enable_memory:
            self.memory = Memory.from_config(memory_config)
            self.memory_tools = MemoryTools(self.memory)
        else:
            self.memory = None
            self.memory_tools = None
        
        ########################################################################
        # TODO: Add tools for the base agent, as well as web search and memory 
        # if they are enabled
        ########################################################################
        # TODO: Add tools for the base agent
        self.tools = [recommend_movies, general_qa, find_time, find_price, find_balance, book_ticket, file_request]

        # enable web search
        if self.web_tools: 
            # TODO: add web search tool to self.tools and delete `pass`
            self.tools.append(self.web_tools.web_search)
        
        # add memory tools if enabled
        if self.memory_tools:
            # TODO: add the relevant memory tools here and delete `pass`
            self.tools.extend([
                self.memory_tools.store_memory,
                self.memory_tools.search_memories,
                self.memory_tools.get_all_memories,
                self.memory_tools.update_memory,
                self.memory_tools.delete_memory,
            ])
       
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        # Initialize ReAct agent
        self.react = dspy.ReAct(
            EnhancedMovieTicketSignature,
            tools=self.tools,
            max_iters=6
        )
    
    def forward(self, user_request: str):
        """Process user input with enhanced capabilities."""
        return self.react(user_request=user_request)

class EnhancedMovieTicketSignature(dspy.Signature):
    """
    You are a movie ticket agent with access to movie recommendation tools,
    ticket-booking tools, web search, and long-term memory.

    Choose the correct tool based on the user's request.

    Rules:
    - If the user shares a preference, favorite, personal fact, or past experience,
      use store_memory.
      Examples:
      - "my favorite genre is sci-fi"
      - "my favorite movie is The Matrix"
      - "I watched my first sci-fi movie when I was 6"
      - "remember that I prefer evening showtimes"

    - If the user asks about something they told you before, use search_memories.
      Examples:
      - "what is my favorite movie?"
      - "what is my favorite genre?"
      - "when did I watch my first sci-fi movie?"

    - If the user asks what you remember about them, use get_all_memories.

    - Use web_search for current or up-to-date movie information such as:
      release dates, actors, directors, upcoming movies, and recent news.

    - Do NOT use general_qa when the answer should come from memory.
    - Do NOT use recommend_movies when the user is simply stating a preference.
    - Once the request is fulfilled, finish and provide a short final answer.
    """
    user_request: str = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc="Final concise answer to the user."
    )


enhanced_agent = EnhancedMovieTicketAgent(enable_web_search=True, enable_memory=True)
