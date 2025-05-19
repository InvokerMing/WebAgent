# WebAgent

WebAgent is an automated web browsing tool powered by AI that can navigate websites, understand page content, and complete tasks based on user instructions.

## Features

- **AI-Powered Visual Understanding**: Uses Google's Gemini API to analyze screenshots and understand webpage content
- **Multiple Operation Modes**: Choose between HTML-only mode (faster) or screenshot+HTML modes (more accurate)
- **Batch Processing**: Capture multiple screenshots before analysis for better context awareness
- **Automated Navigation**: Intelligently clicks, types, selects, and navigates through websites
- **Cookie Consent Handling**: Automatically handles cookie consent popups
- **Scrollable Content Analysis**: Captures content by scrolling through long pages

## Requirements

- Python 3.7+
- Chrome/Chromium browser
- ChromeDriver (compatible with your browser version)
- Google Gemini API key

## Dependencies

```
selenium
beautifulsoup4
google-genai
```

## Installation

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Make sure ChromeDriver is installed and in your PATH
4. Set your Google Gemini API key in the `GEMINI_API_KEY` variable in the code (By creating .env file where writes "GEMINI_API_KEY=YOUR_KEY")

## Usage

Run the script with:

```
python WebAgent.py
```

Then follow the interactive prompts to enter your task instructions.

### Command Options

- `-help`: Display help information
- `-url [new_url]`: Change the target URL
- `-set mode [option]`: Set operation mode
  - `html`: HTML-only mode (no screenshots)
  - `batch`: Batch mode - capture all screenshots before processing (default)
  - `standard`: Process each screenshot immediately
- `-set scrolls [number]`: Set maximum number of page scrolls
- `quit`: Exit the program

### Examples

```
-url https://www.imdb.com
-set mode html
-set scrolls 5
Tell me top3 on IMDb this week
```

## Operation Modes

### HTML-Only Mode

Uses only HTML content for analysis without capturing screenshots. Much faster but potentially less accurate for visually complex pages.

### Batch Mode (Default)

Captures multiple screenshots while scrolling through the page, then analyzes them together for better context understanding.

### Standard Mode

Processes each screenshot immediately after capture. Useful for very long pages where batch processing might hit token limits.

## How It Works

1. Opens the specified URL in a Chrome browser
2. Handles any cookie consent dialogs
3. Captures content through screenshots and/or HTML parsing
4. Analyzes content using Google's Gemini AI models
5. Determines the best actions to fulfill the user's goal
6. Executes actions on the page
7. Repeats the process until an answer is found or maximum steps reached

## Notes

- Screenshots are temporarily stored in the `tmp/` folder and cleaned up after task completion
- The Gemini API key should be kept secure and not committed to public repositories
