import httpx
from mcp.server.fastmcp import FastMCP
from typing import Any, Dict


# FastMCP
mcp = FastMCP(name='converter')


# REVOLUT_API_EUR_PLN = 'https://revolut.com/currency-converter/convert-eur-to-pln-exchange-rate?amount=38757.37'
# REVOLUT_API_PLN_EUR = 'https://revolut.com/currency-converter/convert-pln-to-eur-exchange-rate?amount=164519.28'
GOOGLE_API = 'https://api.exchangerate-api.com/v4/latest'


async def get_request(url: str) -> Dict[str, Any] | None:
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
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {'error': str(e)}


# Tools (specific operations/actions with typed inputs and outputs)
@mcp.tool()
async def google_convert(currency_from: str, currency_to: str, amount: float) -> Dict[str, Any] | None:
    '''
    Converts a specified amount from one currency to another using the current exchange rate.
    Args:
        currency_from: The 3-letter code of the currency to convert from (e.g., 'EUR')
        currency_to: The 3-letter code of the currency to convert to (e.g., 'PLN')
        amount: The numeric amount to convert
    Returns:
        Dictionary containing the exchange rate and converted amount, or error information if conversion fails.
    '''
    try:
        url = f'{GOOGLE_API}/{currency_from}'
        data = await get_request(url)
        exchange_rate = data['rates'][currency_to]
        converted_amount = amount * exchange_rate
        return {
            'exchange_rate': exchange_rate,
            'converted_amount': converted_amount
        }
    except Exception as e:
        return {'error': str(e)}


def main():
    mcp.run()


if __name__ == '__main__':
    main()