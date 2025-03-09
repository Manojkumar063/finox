import streamlit as st
import os
import json
import requests
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Flaix - Financial Assistant",
    page_icon="💰",
    layout="centered"
)

# Initialize Gemini client
api_key = st.secrets["GOOGLE_API_KEY"]
client = genai.Client(api_key=api_key)

# Indian Stock Market API base configuration
INDIAN_API_KEY = st.secrets["FINANCE_KEY"]
INDIAN_API_BASE_URL = "https://stock.indianapi.in"

# Define API endpoints and their parameters
API_ENDPOINTS = {
    "get_stock_details": {
        "endpoint": "/stock",
        "required_params": ["stock_name"],
        "param_mapping": {"stock_name": "name"},
        "description": "Get details for a specific stock"
    },
    "get_trending_stocks": {
        "endpoint": "/trending",
        "required_params": [],
        "param_mapping": {},
        "description": "Get trending stocks in the market"
    },
    "get_market_news": {
        "endpoint": "/news",
        "required_params": [],
        "param_mapping": {},
        "description": "Get latest stock market news"
    },
    "get_mutual_funds": {
        "endpoint": "/mutual_funds",
        "required_params": [],
        "param_mapping": {},
        "description": "Get mutual funds data"
    },
    "get_ipo_data": {
        "endpoint": "/ipo",
        "required_params": [],
        "param_mapping": {},
        "description": "Get IPO data"
    },
    "get_bse_most_active": {
        "endpoint": "/BSE_most_active",
        "required_params": [],
        "param_mapping": {},
        "description": "Get BSE most active stocks"
    },
    "get_nse_most_active": {
        "endpoint": "/NSE_most_active",
        "required_params": [],
        "param_mapping": {},
        "description": "Get NSE most active stocks"
    },
    "get_historical_data": {
        "endpoint": "/historical_data",
        "required_params": ["stock_name"],
        "optional_params": ["period"],
        "default_values": {"period": "1m", "filter": "default"},
        "param_mapping": {},
        "description": "Get historical data for a stock"
    }
}

# Unified API call function
def call_indian_api(endpoint, params=None):
    """
    Generic function to call the Indian Stock Market API
    
    Args:
        endpoint: API endpoint suffix (e.g., '/stock', '/trending')
        params: Optional parameters for the API call
        
    Returns:
        JSON response from the API
    """
    url = f"{INDIAN_API_BASE_URL}{endpoint}"
    headers = {"X-Api-Key": INDIAN_API_KEY}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# Function to call API by name
def call_api_by_name(api_name, **kwargs):
    """
    Call an API by its name from the API_ENDPOINTS dictionary
    
    Args:
        api_name: Name of the API to call (key in API_ENDPOINTS)
        **kwargs: Parameters to pass to the API
        
    Returns:
        JSON response from the API
    """
    if api_name not in API_ENDPOINTS:
        return {"error": f"Unknown API: {api_name}"}
    
    api_info = API_ENDPOINTS[api_name]
    endpoint = api_info["endpoint"]
    
    # Check required parameters
    for param in api_info.get("required_params", []):
        if param not in kwargs:
            return {"error": f"Missing required parameter: {param}"}
    
    # Apply parameter mapping
    mapped_params = {}
    for param, value in kwargs.items():
        mapped_name = api_info.get("param_mapping", {}).get(param, param)
        mapped_params[mapped_name] = value
    
    # Apply default values
    for param, value in api_info.get("default_values", {}).items():
        if param not in mapped_params:
            mapped_params[param] = value
    
    return call_indian_api(endpoint, mapped_params)

# Improved orchestrator function
def orchestrator(query):
    """
    Determines if the query requires market data and which API to call
    Returns: (needs_api, api_function, params)
    """
    # Create a more precise prompt for the orchestrator
    orchestrator_prompt = """
    You are an orchestrator for a financial assistant specialized in Indian markets. Your job is to analyze user queries and determine if they need real-time market data.

    IMPORTANT: Be very precise in your analysis. Only return TRUE for "needs_api" when the query EXPLICITLY asks for current market data, stock prices, or listings.

    Examples where needs_api should be TRUE:
    - "Show me the most active stocks on NSE today" → get_nse_most_active
    - "What is the current price of Reliance?" → get_stock_details with stock_name="Reliance"
    - "Tell me about trending stocks" → get_trending_stocks
    - "What are the latest IPOs?" → get_ipo_data

    Examples where needs_api should be FALSE:
    - "What is compound interest?"
    - "How should I start investing?"
    - "What are the tax benefits of PPF?"
    - "Explain mutual funds to me"

    Available API functions:
    - get_stock_details(stock_name): Get details for a specific stock
    - get_trending_stocks(): Get trending stocks in the market
    - get_market_news(): Get latest stock market news
    - get_mutual_funds(): Get mutual funds data
    - get_ipo_data(): Get IPO data
    - get_bse_most_active(): Get BSE most active stocks
    - get_nse_most_active(): Get NSE most active stocks
    - get_historical_data(stock_name, period="1m"): Get historical data for a stock

    User query: """ + query + """

    Respond in JSON format with the following structure:
    {
        "needs_api": true/false,
        "function": "function_name_if_needed",
        "params": {
            "param1": "value1",
            "param2": "value2"
        }
    }
    """
        
    # Create content for the orchestrator
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=orchestrator_prompt)
            ],
        ),
    ]
    
    # Configure generation parameters
    generate_content_config = types.GenerateContentConfig(
        temperature=0.2,
        top_p=0.95,
        top_k=40,
        max_output_tokens=500,
        response_mime_type="text/plain",
    )
    
    # Generate content
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=contents,
        config=generate_content_config,
    )
    
    # Parse the response
    try:
        decision_text = response.text
        # Extract JSON from the response (it might be wrapped in markdown code blocks)
        if "```json" in decision_text:
            json_str = decision_text.split("```json")[1].split("```")[0].strip()
        elif "```" in decision_text:
            json_str = decision_text.split("```")[1].strip()
        else:
            json_str = decision_text.strip()
        
        decision = json.loads(json_str)
        return decision
    except Exception as e:
        print(f"Error parsing orchestrator response: {e}")
        return {"needs_api": False}

# Language setting

# Financial assistant system prompt
SYSTEM_PROMPT = f"""You are Flaix, a helpful and knowledgeable financial assistant designed specifically for Indian users. Your purpose is to improve financial literacy and provide guidance on investments in the Indian market.

Key responsibilities:
1. Explain financial concepts in simple, easy-to-understand language
2. Provide information about different investment options available in India (stocks, mutual funds, bonds, PPF, FDs, etc.)
3. Help users understand investment risks and returns
4. Explain tax implications of different investments in the Indian context
5. Guide users on how to start investing based on their goals and risk tolerance
6. Answer questions about market trends and financial news in India
"""

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "content": SYSTEM_PROMPT},
        {"role": "model", "content": "Hello! I am Flaix, your financial assistant. You can ask me about investments, financial planning, or any other financial topic."}
    ]

# App title and description
st.title("Flaix - Your Financial Assistant")
st.markdown("Ask any questions about investing, financial planning, or the Indian financial market.")

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user" and message["content"] != SYSTEM_PROMPT:
        with st.chat_message("user"):
            st.write(message["content"])
    elif message["role"] == "model":
        with st.chat_message("assistant"):
            st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about finance or investing..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # First, use the orchestrator to determine if we need to call an API
            decision = orchestrator(prompt)
            
            # If we need to call an API, do so and add the result to the context
            api_context = ""
            if decision.get("needs_api", False):
                function_name = decision.get("function", "")
                params = decision.get("params", {})
                
                message_placeholder.write("Fetching real-time market data...")
                
                if function_name in API_ENDPOINTS:
                    api_result = call_api_by_name(function_name, **params)
                    api_context = f"\nHere is the real-time market data from the Indian Stock Market API:\n{json.dumps(api_result, indent=2)}\n\nPlease use this data to provide an informative response to the user's query."
            
            
            # Prepare the user query with API context if available
            user_query = prompt
            if api_context:
                user_query = f"{prompt}\n\n[SYSTEM NOTE: {api_context}]"
            
            # Prepare the system message
            system_message = SYSTEM_PROMPT
            if len(st.session_state.messages) > 2:  # If we have conversation history
                # Extract previous conversation for context
                conversation_history = ""
                for i in range(1, min(5, len(st.session_state.messages) - 1)):  # Get up to 5 previous exchanges
                    if st.session_state.messages[i]["role"] == "user" and st.session_state.messages[i]["content"] != SYSTEM_PROMPT:
                        conversation_history += f"User: {st.session_state.messages[i]['content']}\n"
                    elif st.session_state.messages[i]["role"] == "model":
                        conversation_history += f"Assistant: {st.session_state.messages[i]['content']}\n"
                
                system_message += f"\n\nPrevious conversation:\n{conversation_history}"
            
            # Create content for the LLM
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=system_message)
                    ],
                ),
                types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text(text="I understand my role as Flaix, a financial assistant for Indian users. I'll provide helpful information about investing and financial planning in simple language.")
                    ],
                ),
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=user_query)
                    ],
                ),
            ]
            
            # Configure generation parameters
            generate_content_config = types.GenerateContentConfig(
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
                response_mime_type="text/plain",
            )
            
            # Stream the response
            response_stream = client.models.generate_content_stream(
                model="gemini-1.5-flash",
                contents=contents,
                config=generate_content_config,
            )
            
            # Process streaming response
            for chunk in response_stream:
                if hasattr(chunk, 'text'):
                    full_response += chunk.text
                    message_placeholder.write(full_response + "▌")
            
            # Final update without cursor
            message_placeholder.write(full_response)
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            full_response = "I'm sorry, I encountered an error. Please try again later."
            message_placeholder.write(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "model", "content": full_response})
