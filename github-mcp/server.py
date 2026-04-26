import httpx
from mcp.server.fastmcp import FastMCP
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData, INVALID_PARAMS, INTERNAL_ERROR

# Create the MCP server - give it a name
mcp = FastMCP("github-stats")

GITHUB_API = "https://api.github.com"

@mcp.tool()
async def get_github_user(username: str) -> str:
    """
    Get public profile information for a GitHub user.

    Args:
        username: The GitHub username to look up (e.g. 'torvalds')
    """
    if not username or not username.strip():
        raise McpError(
            ErrorData(
                code=INVALID_PARAMS,
                message="Username cannot be empty."
            )
        )

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(
                f"{GITHUB_API}/users/{username.strip()}",
                headers={"Accept": "application/vnd.github.v3+json"}
            )
        except httpx.TimeoutException:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message="GitHub API timed out. Try again in a moment."
                )
            )

        if response.status_code == 404:
            raise McpError(
                ErrorData(
                    code=INVALID_PARAMS,
                    message=f"GitHub user '{username}' does not exist."
                )
            )

        if response.status_code == 403:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message="GitHub rate limit reached. Wait 60 seconds or add a GITHUB_TOKEN."
                )
            )

        if response.status_code != 200:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message=f"GitHub API returned {response.status_code}."
                )
            )

        data = response.json()

        return (
            f"Name: {data.get('name', 'Not set')}\n"
            f"Bio: {data.get('bio', 'No bio')}\n"
            f"Followers: {data.get('followers', 0):,}\n"
            f"Public Repos: {data.get('public_repos', 0)}\n"
            f"Location: {data.get('location', 'Not specified')}\n"
            f"Profile: {data.get('html_url')}"
        )
    
@mcp.tool()
async def get_github_repos(username: str, limit: int = 5) -> str:
    """
    Get the most recently updated public repositories for a GitHub user.

    Args:
        username: The GitHub username
        limit: How many repos to return (default 5, max 10)
    """
    limit = min(limit, 10)  # Safety cap

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{GITHUB_API}/users/{username}/repos",
            params={"sort": "updated", "per_page": limit},
            headers={"Accept": "application/vnd.github.v3+json"}
        )

        if response.status_code != 200:
            return f"Could not fetch repos for '{username}'."

        repos = response.json()

        if not repos:
            return f"No public repositories found for '{username}'."

        lines = [f"Top {len(repos)} repos for @{username}:\n"]

        for repo in repos:
            stars = repo.get("stargazers_count", 0)
            lang = repo.get("language") or "No language"
            lines.append(
                f"• {repo['name']} "
                f"({lang}, ⭐ {stars:,})\n"
                f"  {repo.get('description') or 'No description'}"
            )

        return "\n".join(lines)

if __name__ == "__main__":
    mcp.run(transport="stdio")