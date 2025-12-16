import httpx
from mcp.server.fastmcp import FastMCP
from typing import Any, Dict


# FastMCP
mcp = FastMCP(name='Revolut', json_response=True)


REVOLUT_API_EUR_PLN = 'https://www.revolut.com/currency-converter/convert-eur-to-pln-exchange-rate?amount=38757.37'
REVOLUT_API_PLN_EUR = 'https://www.revolut.com/currency-converter/convert-pln-to-eur-exchange-rate?amount=164519.28'


# Tools (specific operation with typed inputs and outputs)
async def make_get_request(url: str) -> Dict[str, Any] | None:
    '''
    Make a GET request to API
    Args:
        url: The URL to make the request to
    Returns:
        The data from the request
    '''
    headers = {
        'Accept': 'application/json'
    }
    async with httpx.AsyncHTTPClient(follow_redirects=True) as client:
        try:
            response = await client.get(url, headers=headers, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {'error': str(e)}


@mcp.tool()
# async def get_revolut_exchange_rate(amount: float, currency: str) -> str:
async def get_revolut_exchange_rate() -> str:
    '''
    Get the exchange rate for a given amount and currency
    Args:
        amount: The amount to convert
        currency: The currency to convert to
    Returns:
        The exchange rate
    '''
    url = REVOLUT_API_EUR_PLN if currency == 'EUR' else REVOLUT_API_PLN_EUR
    # data = await make_get_request(url)
    data = await make_get_request(REVOLUT_API_EUR_PLN)
    return data


def main():
    mcp.run(transport='streamable-http')
    # mcp.run(transport='stdio')


if __name__ == '__main__':
    main()