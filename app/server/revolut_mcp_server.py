import requests
from mcp.server.fastmcp import FastMCP
from typing import Any, Dict


# FastMCP
mcp = FastMCP(name='Revolut')


# REVOLUT_API_EUR_PLN = 'https://revolut.com/currency-converter/convert-eur-to-pln-exchange-rate?amount=38757.37'
# REVOLUT_API_PLN_EUR = 'https://revolut.com/currency-converter/convert-pln-to-eur-exchange-rate?amount=164519.28'
GOOGLE_API = 'https://api.exchangerate-api.com/v4/latest'


def make_get_request(url: str) -> Dict[str, Any] | None:
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
    try:
        response = requests.get(url, headers=headers, timeout=10.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {'error': str(e)}


# Tools (specific operations/actions with typed inputs and outputs)
@mcp.tool()
def get_revolut_exchange_rate(currency: str, amount: float) -> Dict[str, Any] | None:
    '''
    Get the exchange rate for a given amount and currency
    Args:
        amount: The amount to convert
        currency: The currency to convert to
    Returns:
        The exchange rate
    '''
    url = f'{GOOGLE_API}/EUR'
    data = make_get_request(url)
    exchange_rate = data['rates'][currency]
    converted_amount = amount * exchange_rate
    return {
        'exchange_rate': exchange_rate,
        'converted_amount': converted_amount
    }


if __name__ == '__main__':
    mcp.run()