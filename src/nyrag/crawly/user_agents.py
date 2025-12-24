USER_AGENTS = {
    "chrome": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    ),
    "firefox": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) " "Gecko/20100101 Firefox/121.0"),
    "safari": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.2 Safari/605.1.15"
    ),
    "mobile": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1"
    ),
    "bot": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
}


def get_user_agent(agent_type="chrome"):
    """Get user agent string by type."""
    return USER_AGENTS.get(agent_type.lower(), USER_AGENTS["chrome"])
