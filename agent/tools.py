"""Custom function tools for the agent."""

CUSTOM_TOOLS: list[dict] = [
    {
        "type": "function",
        "name": "navigate",
        "description": (
            "Navigate directly to a URL when you know the destination and want to open it "
            "faster than typing into the browser UI."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL that starts with http:// or https://",
                },
            },
            "required": ["url"],
        },
    },
    {
        "type": "function",
        "name": "search_web",
        "description": (
            "Search the web in one step and open a results page for the query. "
            "Use this when you need to discover a website or find public information."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text",
                },
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "go_back",
        "description": (
            "Go back one page in the current browser tab. "
            "Use this to recover from the wrong page, wrong restaurant, wrong product, or an empty result state."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "type": "function",
        "name": "observe",
        "description": (
            "Inspect the visible page and return a structured list of visible interactive elements "
            "with ids, labels, roles, and screen coordinates. Use this before choosing among many "
            "similar buttons, links, menu items, cards, or input fields."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "Short description of what kind of element you want to find on the current page",
                },
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "click_observed",
        "description": (
            "Click a visible element returned by the latest observe() call. "
            "Use this when you want a deterministic click on a specific labeled item."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The element id returned by observe()",
                },
            },
            "required": ["element_id"],
        },
    },
    {
        "type": "function",
        "name": "type_into_observed",
        "description": (
            "Focus a visible input-like element returned by the latest observe() call and type text into it. "
            "Use this instead of blind coordinate typing when the correct field is visible."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The element id returned by observe()",
                },
                "text": {
                    "type": "string",
                    "description": "The text to type",
                },
                "replace": {
                    "type": "boolean",
                    "description": "Whether to select existing text before typing. Defaults to true.",
                },
            },
            "required": ["element_id", "text"],
        },
    },
    {
        "type": "function",
        "name": "submit_observed_search",
        "description": (
            "After typing into a search or filter field, commit the query with Enter on that same observed element. "
            "Use when the UI needs explicit submission and a visible search button is missing or unreliable. "
            "Prefer this over calling done() or navigating away while results have not loaded."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "The same element_id you used with type_into_observed for this search field",
                },
            },
            "required": ["element_id"],
        },
    },
    {
        "type": "function",
        "name": "done",
        "description": (
            "Finish the task and return the final outcome to the user. "
            "The message must contain concrete details, not a plan."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Final answer with concrete outcome, in the user's language",
                },
            },
            "required": ["message"],
        },
    },
    {
        "type": "function",
        "name": "ask_user",
        "description": (
            "Ask the user for missing information or confirmation before risky or irreversible actions. "
            "Also use this for login, CAPTCHA, payment, deletion, or sensitive submissions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Clear question to the user, in their language",
                },
            },
            "required": ["question"],
        },
    },
]
