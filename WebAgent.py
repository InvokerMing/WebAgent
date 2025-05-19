from google import genai
import json
import time
from pathlib import Path

from bs4 import BeautifulSoup
import re
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.common.exceptions import NoSuchElementException, WebDriverException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from typing import List, Dict, Any, Optional
from selenium.webdriver.support.ui import Select

# API configuration
GEMINI_API_KEY = "AIzaSyDKcjYKwk2F506Bk1DnmgmnsJbl9wgvW0c"
VISION_MODEL_NAME = "gemini-2.0-flash"
TEXT_MODEL_NAME = "gemini-2.0-flash"
TARGET_URL = "https://www.imdb.com/"

try:
    GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"Error configuring Gemini API: {e}. Please ensure GEMINI_API_KEY is set correctly.")
    exit()

# Perception Module
class PerceptionModule:
    """Handles screenshot analysis and interprets visual elements of web pages"""
    def __init__(self):
        self.model = GEMINI_CLIENT.models

    def perceive(self, screenshot_path: str, simplified_html: Optional[str] = None) -> Dict[str, Any]:
        """Process a single screenshot and return structured perception data."""
        print(f"Perception Module: Analyzing screenshot {screenshot_path}...")
        if not Path(screenshot_path).is_file():
            return {"error": f"Screenshot file not found: {screenshot_path}"}

        prompt_text = self._get_perception_prompt(simplified_html)
        
        try:
            print("Perception Module: Uploading screenshot...")
            uploaded_file = None
            uploaded_file = GEMINI_CLIENT.files.upload(file=screenshot_path)
            print(f"Perception Module: Screenshot uploaded successfully (URI: {uploaded_file.uri}).")

            print("Perception Module: Calling Gemini Vision API...")
            response = self.model.generate_content(
                model=VISION_MODEL_NAME,
                contents=[uploaded_file, prompt_text]
            )
            print("Perception Module: Received response from Gemini Vision API.")

            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            structured_state = json.loads(response_text)
            print("Perception Module: State parsed successfully.")
            if not isinstance(structured_state, dict) or "interactive_elements" not in structured_state:
                 raise ValueError("Parsed JSON does not have the expected 'interactive_elements' structure.")

            return structured_state

        except Exception as e:
            print(f"Perception Module: Error during perception: {e}")
            return {"error": str(e)}
        finally:
            if 'uploaded_file' in locals() and uploaded_file and uploaded_file.name:
                try:
                    print(f"Perception Module: Deleting uploaded file {uploaded_file.name}...")
                    GEMINI_CLIENT.files.delete(name=uploaded_file.name)
                    print("Perception Module: Uploaded file deleted.")
                except Exception as delete_error:
                    print(f"Perception Module: Could not delete file {uploaded_file.name} after processing: {delete_error}")
    
    def perceive_batch(self, screenshot_paths: List[str], simplified_html: Optional[str] = None) -> List[Dict[str, Any]]:
        """Process multiple screenshots in a single batch to get combined perception."""
        print(f"Perception Module: Batch analyzing {len(screenshot_paths)} screenshots...")
        
        # Check if files exist
        missing_files = [path for path in screenshot_paths if not Path(path).is_file()]
        if missing_files:
            print(f"Error: {len(missing_files)} screenshot files not found")
            return [{"error": f"Screenshot files not found: {missing_files}"}]
            
        prompt_text = self._get_perception_prompt(simplified_html)
        
        try:
            # Upload all screenshots and keep track of them
            uploaded_files = []
            for path in screenshot_paths:
                print(f"Uploading screenshot: {path}...")
                uploaded_file = GEMINI_CLIENT.files.upload(file=path)
                uploaded_files.append(uploaded_file)
                print(f"Uploaded successfully (URI: {uploaded_file.uri})")
            
            # Prepare contents with all uploaded files plus the prompt
            contents = uploaded_files + [prompt_text]
            
            print(f"Perception Module: Calling Gemini Vision API with {len(uploaded_files)} screenshots...")
            response = self.model.generate_content(
                model=VISION_MODEL_NAME,
                contents=contents
            )
            print("Perception Module: Received batch response from Gemini Vision API.")
            
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
                
            structured_state = json.loads(response_text)
            print("Perception Module: Batch state parsed successfully.")
            
            # For batch processing, we'll return the result as a single-item list
            # to maintain compatibility with the existing code that expects a list
            return [structured_state]
            
        except Exception as e:
            print(f"Perception Module: Error during batch perception: {e}")
            return [{"error": f"Batch perception error: {str(e)}"}]
        finally:
            # Clean up all uploaded files
            for uploaded_file in uploaded_files:
                if uploaded_file and uploaded_file.name:
                    try:
                        print(f"Deleting uploaded file {uploaded_file.name}...")
                        GEMINI_CLIENT.files.delete(name=uploaded_file.name)
                    except Exception as delete_error:
                        print(f"Could not delete file {uploaded_file.name}: {delete_error}")
                        
    def _get_perception_prompt(self, simplified_html: Optional[str] = None) -> str:
        """Generate the prompt for perception tasks."""
        return f"""
        Analyze this webpage screenshot. Considering the optional simplified HTML info below (if provided), perform these tasks:
        1. Briefly summarize the main content and purpose of the page.
        2. Identify and list the main *interactive* elements (buttons, links, inputs, selects, etc.). For each, provide:
           - A temporary unique ID (e.g., 'element_1', 'element_2'). This ID will be used to refer to this element.
           - The element type (e.g., 'button', 'link', 'input', 'select').
           - The visible text or its function (e.g., 'Login', 'Search movies'). If no text, describe its purpose (e.g. 'Search icon button').
           - A visual description or location (e.g., 'Red button top-right', 'Input field below logo').
           - **CRITICAL: A precise CSS Selector OR an XPath for this element. Provide the most robust one you can determine. If you absolutely cannot determine a reliable locator, set both 'css_selector' and 'xpath' to null or "locator_unavailable". Prioritize CSS selectors if possible.**
        3. Identify and list key *non-interactive content* elements relevant to the likely user goal (e.g., product names, prices, article titles, drama titles, ratings that are NOT directly clickable links/buttons themselves). For each, provide:
           - A type describing the content (e.g., 'drama_title', 'rating', 'product_price').
           - The text content of the element.
        4. Based on the page content and identified elements, suggest some possible user actions related to the goal.
        5. IMPORTANT: Structure your entire response as a single JSON object. Ensure all string values within the JSON are properly escaped. Example format:
           {{
             "summary": "A movie listing page.",
             "interactive_elements": [
               {{"id": "element_1", "type": "button", "text": "Login", "visual_description": "Blue button top right", "css_selector": "#loginButton", "xpath": null}},
               {{"id": "element_2", "type": "link", "text": "More info", "visual_description": "Link below image", "css_selector": "a.more-info", "xpath": "//a[contains(text(),'More info')]"}},
               {{"id": "element_3", "type": "input", "text": "Search movies", "visual_description": "Input field at top", "css_selector": "input[name='q']", "xpath": null}},
               {{"id": "element_4", "type": "button", "text": null, "visual_description": "Magnifying glass icon for search", "css_selector": "button.search-icon", "xpath": null}}
             ],
             "content_elements": [
                {{"type": "drama_title", "text": "Example Drama 1"}},
                {{"type": "rating", "text": "8.5"}}
             ],
             "potential_actions": ["Click 'Login'", "Type 'action movie' into 'Search movies' input"]
           }}

        Simplified HTML (if available):
        {simplified_html if simplified_html else 'None'}
        """


# Planning & Reasoning Module
class PlanningReasoningModule:
    def __init__(self):
        self.model = GEMINI_CLIENT.models
        
    def analyze_html_only(self, goal: str, html_content: str, current_url: str, 
                         history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze HTML content only (no screenshots) to generate action plan
        """
        print("\n=== Planning Module: HTML-Only Analysis ===")
        
        if not history:
            history = []
            
        # Preprocess HTML to make it more manageable
        processed_html = self._preprocess_html(html_content)
        print(f"HTML content length after preprocessing: {len(processed_html)} characters")
        
        # Format history
        formatted_history = self._format_history(history)
        
        prompt = f"""
        You are a web assistant. Your task is to analyze HTML content from a webpage to address the user's goal.
        
        User Goal: {goal}
        Current URL: {current_url}
        
        === HTML Content ===
        {processed_html[:200000]}  # Using more content since we don't have screenshots
        
        === Action History ===
        {formatted_history}
        
        Based on the HTML content, thoroughly analyze the page to either:
        1. Provide a direct ANSWER to the user's goal if the information is available in the content
        2. Determine the next best ACTION to take if further information is needed or interaction is required
        
        Your response MUST be one of the following JSON formats:
        1. Direct Answer Format:
        {{"action_type": "ANSWER", "content": "Detailed answer to the user's goal based on the available information", "scroll_position": 0}}
        
        2. Click Element Format:
        {{"action_type": "click", "element_id": "css_selector_or_xpath", "comment": "Reason for clicking this element"}}
        
        3. Type Text Format:
        {{"action_type": "type", "element_id": "css_selector_or_xpath", "text": "text_to_type", "comment": "Reason for typing this"}}
        
        4. Select Option Format:
        {{"action_type": "select", "element_id": "css_selector_or_xpath", "option_text": "text_of_option_to_select", "comment": "Reason for selecting"}}
        
        5. Navigate Format:
        {{"action_type": "navigate", "url": "target_url", "comment": "Reason for navigating, use sparingly"}}
        
        6. Stop/Finished Format:
        {{"action_type": "stop", "reason": "Explain why (e.g., goal impossible to achieve, stuck in a loop)"}}
        
        IMPORTANT: In HTML-only mode, provide element_id as either a CSS selector or XPath that can be used directly 
        to locate the element, since there are no screenshot-based element IDs available.
        
        IMPORTANT: If you find the answer, set "scroll_position" to 0 in your ANSWER response, as we're analyzing the full page.
        
        Output ONLY the JSON for your response without any additional text or explanation.
        """
        
        try:
            response = self.model.generate_content(model=TEXT_MODEL_NAME, contents=prompt)
            response_text = response.text.strip()
            if response_text.startswith("```json"): response_text = response_text[7:-3]
            elif response_text.startswith("```"): response_text = response_text[3:-3]
            
            result = json.loads(response_text)
            print(f"Planning Module: HTML-Only analysis complete - {result.get('action_type', 'unknown')}")
            return result
            
        except Exception as e:
            print(f"Planning Module: Error during HTML-only analysis: {e}")
            return {'action_type': 'error', 'message': f'HTML analysis error: {e}'}

    def analyze_combined_data(self, goal: str, current_states: List[Dict[str, Any]], 
                             html_content: str, current_url: str, 
                             history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze combined screenshot data and HTML to generate a direct answer or action plan.
        """
        print("\n=== Planning Module: Analyzing Combined Data ===")
        
        if not history:
            history = []
            
        # Preprocess HTML to make it more manageable
        processed_html = self._preprocess_html(html_content)
        print(f"HTML content length after preprocessing: {len(processed_html)} characters")
        
        # Extract scroll positions information for batch mode
        scroll_positions = []
        scroll_indexes = []
        
        # If we have batch mode data (a single state with multiple positions)
        if len(current_states) == 1 and "scroll_positions" in current_states[0]:
            scroll_positions = current_states[0].get("scroll_positions", [])
            scroll_indexes = current_states[0].get("scroll_indexes", list(range(len(scroll_positions))))
            print(f"Found batch mode data with {len(scroll_positions)} scroll positions")
            
        # Format current states from screenshots (assuming we have multiple screenshots from scrolling)
        formatted_states = []
        for i, state in enumerate(current_states):
            summary = state.get('summary', 'No summary available')
            interactive_elements = self._format_elements(state.get('interactive_elements', []))
            content_elements = self._format_content(state.get('content_elements', []))
            
            # Handle scroll position info for both batch and standard mode
            if "scroll_position" in state:
                scroll_position = state.get('scroll_position', 'Unknown')
                scroll_index = state.get('scroll_index', i)
            elif scroll_positions and i < len(scroll_positions):
                scroll_position = scroll_positions[i]
                scroll_index = scroll_indexes[i] if i < len(scroll_indexes) else i
            else:
                scroll_position = 'Unknown'
                scroll_index = i
            
            formatted_states.append(f"Screenshot {i+1} (Scroll {scroll_index}, Position {scroll_position}):\nSummary: {summary}\nInteractive Elements:\n{interactive_elements}\nContent Elements:\n{content_elements}\n")
        
        formatted_states_text = "\n".join(formatted_states)
        
        # Format history
        formatted_history = self._format_history(history)
        
        prompt = f"""
        You are a web assistant. Your task is to analyze multiple screenshots and HTML content from a webpage to address the user's goal.
        
        User Goal: {goal}
        Current URL: {current_url}
        
        === Information from Multiple Screenshots ===
        {formatted_states_text}
        
        === Processed HTML Content ===
        {processed_html[:100000]}  # Limit HTML content length to avoid token limit issues
        
        === Action History ===
        {formatted_history}
        
        Based on the combined information from the screenshots and HTML content, thoroughly analyze the page to either:
        1. Provide a direct ANSWER to the user's goal if the information is available in the current content
        2. Determine the next best ACTION to take if further information is needed or interaction is required
        
        Your response MUST be one of the following JSON formats:
        1. Direct Answer Format:
        {{"action_type": "ANSWER", "content": "Detailed answer to the user's goal based on the available information", "scroll_index": 0}}
        
        2. Click Element Format:
        {{"action_type": "click", "element_id": "element_id_from_screenshots", "comment": "Reason for clicking this element"}}
        
        3. Type Text Format:
        {{"action_type": "type", "element_id": "input_element_id", "text": "text_to_type", "comment": "Reason for typing this"}}
        
        4. Select Option Format:
        {{"action_type": "select", "element_id": "select_element_id", "option_text": "text_of_option_to_select", "comment": "Reason for selecting"}}
        
        5. Navigate Format:
        {{"action_type": "navigate", "url": "target_url", "comment": "Reason for navigating, use sparingly"}}
        
        6. Stop/Finished Format:
        {{"action_type": "stop", "reason": "Explain why (e.g., goal impossible to achieve, stuck in a loop)"}}
        
        IMPORTANT: When providing an ANSWER, include the "scroll_index" field indicating which screenshot or section
        contains the information for your answer. Valid values are 0 to {len(current_states)-1 if len(current_states) > 1 else (len(scroll_positions)-1 if scroll_positions else 0)}.
        This helps the user locate the relevant content on the page.
        
        Output ONLY the JSON for your response without any additional text or explanation. 
        If an element from the screenshots has "locator_unavailable" for both css_selector and xpath, 
        avoid choosing it for click/type/select unless absolutely no other option exists.
        
        Prioritize providing direct ANSWERS when the information is already present in the content. 
        Only suggest actions when more information is needed or interaction is required.
        """
        
        try:
            response = self.model.generate_content(model=TEXT_MODEL_NAME, contents=prompt)
            response_text = response.text.strip()
            if response_text.startswith("```json"): response_text = response_text[7:-3]
            elif response_text.startswith("```"): response_text = response_text[3:-3]
            
            result = json.loads(response_text)
            print(f"Planning Module: Analysis complete - {result.get('action_type', 'unknown')}")
            return result
            
        except Exception as e:
            print(f"Planning Module: Error analyzing combined data: {e}")
            return {'action_type': 'error', 'message': f'Analysis error: {e}'}

    def _preprocess_html(self, html_content: str, max_length: int = 1000000) -> str:
        """Cleans and reduces HTML content by removing scripts and non-essential elements"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            for element in soup(['script', 'style', 'meta', 'link', 'header', 'footer', 'nav', 'aside', 'form']):
                element.decompose()
            for comment in soup.find_all(text=lambda text: isinstance(text, str) and text.strip().startswith('<!--')):
                comment.extract()

            # Keep only interactable elements and their direct text content, or meaningful content tags
            body = soup.find('body')
            if body:
                # Find all links, buttons, inputs, selects, textareas
                important_tags = body.find_all(['a', 'button', 'input', 'select', 'textarea', 'label', 'h1', 'h2', 'h3', 'p', 'li', 'span'])
                new_body = soup.new_tag('body')
                for tag in important_tags:
                    # Keep relevant attributes
                    attrs_to_keep = ['id', 'class', 'name', 'type', 'value', 'href', 'placeholder', 'aria-label', 'role', 'title']
                    new_attrs = {k: v for k, v in tag.attrs.items() if k in attrs_to_keep}
                    cloned_tag = soup.new_tag(tag.name, attrs=new_attrs)
                    cloned_tag.string = re.sub(r'\s+', ' ', tag.get_text(separator=' ', strip=True)) # Consolidate text
                    if cloned_tag.string or tag.name in ['input', 'select', 'textarea', 'button']: # Keep if it has text or is inherently interactive
                        new_body.append(cloned_tag)
                soup.body.replace_with(new_body)
            
            processed_html = str(soup.body if soup.body else soup) # Get only body content if available
            processed_html = re.sub(r'\s+', ' ', processed_html).strip() # Further consolidate whitespace

            if len(processed_html) > max_length:
                end_tag_pos = processed_html[:max_length].rfind('>')
                if end_tag_pos != -1:
                    processed_html = processed_html[:end_tag_pos + 1]
                else:
                    processed_html = processed_html[:max_length]
            return processed_html
        except Exception as e:
            print(f"HTML preprocessing failed: {e}. Returning severely truncated original HTML.")
            return html_content[:1000] + "... (preprocessing error, truncated)"

    def _format_elements(self, elements: List[Dict[str, Any]]) -> str:
        if not elements: return "None"
        formatted = []
        for el in elements:
            text_or_desc = el.get('text') or el.get('visual_description') or 'No text/desc'
            locator_info = ""
            if el.get('css_selector') and el.get('css_selector') != "locator_unavailable":
                locator_info += f", CSS: '{el['css_selector']}'"
            if el.get('xpath') and el.get('xpath') != "locator_unavailable":
                locator_info += f", XPath: '{el['xpath']}'"
            if not locator_info and (el.get('css_selector') == "locator_unavailable" or el.get('xpath') == "locator_unavailable"):
                locator_info = ", Locator: UNAVAILABLE"

            formatted.append(f"- ID: {el.get('id', 'Unknown')}, Type: {el.get('type', 'Unknown')}, Text/Desc: '{text_or_desc}'{locator_info}")
        return "\n".join(formatted)

    def _format_content(self, elements: List[Dict[str, Any]]) -> str:
        if not elements: return "None"
        return "\n".join([f"- Type: {el.get('type', 'Unknown')}, Content: '{el.get('text', 'N/A')}'" for el in elements])

    def _format_history(self, history: List[Dict[str, Any]]) -> str:
        if not history: return "None (first step)"
        formatted = []
        for i, entry in enumerate(history):
            action_str = json.dumps(entry.get('action', {}))
            formatted.append(f"Step {i+1}: URL='{entry.get('url', 'N/A')}', Summary='{entry.get('summary', 'N/A')}', Action={action_str}")
        return "\n".join(formatted)


# Execution Module
class ExecutionModule:
    """Executes actions on the webpage based on planning decisions"""
    def __init__(self, driver: webdriver.Remote):
        self.driver = driver

    def execute(self, action: Dict[str, Any], state: Dict[str, Any]) -> bool:
        action_type = action.get('action_type')
        print(f"Execution Module: Attempting action - {action_type}, Details: {json.dumps(action)}")

        try:
            if action_type == "click":
                element = self._find_element_for_action(action.get('element_id'), state, action.get("comment"))
                if element:
                    print(f"  Clicking element: {action.get('element_id')}")
                    try:
                        element.click()
                    except WebDriverException as e:
                        print(f"  Attempting click via JavaScript due to: {e}")
                        self.driver.execute_script("arguments[0].click();", element)
                    return True
                else:
                    print(f"  Error: Could not find element to click with ID: {action.get('element_id')}")
                    return False
            elif action_type == "type":
                element = self._find_element_for_action(action.get('element_id'), state, action.get("comment"))
                text_to_type = action.get('text', '')
                if element:
                    print(f"  Typing '{text_to_type}' into element {action.get('element_id')}")
                    try:
                        element.clear() # Clear first
                        element.send_keys(text_to_type)
                    except WebDriverException as e:
                        print(f"  Error during send_keys: {e}. Trying JS to set value.")
                        self.driver.execute_script("arguments[0].value = arguments[1];", element, text_to_type)
                        # Dispatch input event if needed for some frameworks
                        self.driver.execute_script("arguments[0].dispatchEvent(new Event('input', { bubbles: true }));", element)
                        self.driver.execute_script("arguments[0].dispatchEvent(new Event('change', { bubbles: true }));", element)

                    return True
                else:
                    print(f"  Error: Could not find element to type into with ID: {action.get('element_id')}")
                    return False
            elif action_type == "select":
                element_id = action.get('element_id')
                option_text = action.get('option_text')
                element = self._find_element_for_action(element_id, state, action.get("comment"))
                if element and element.tag_name.lower() == 'select':
                    select_obj = Select(element)
                    try:
                        select_obj.select_by_visible_text(option_text)
                        print(f"    Successfully selected option '{option_text}'.")
                        return True
                    except NoSuchElementException:
                        print(f"  Warn: Option '{option_text}' not found by visible text. Trying by value.")
                        try:
                            select_obj.select_by_value(option_text)
                            print(f"    Successfully selected option by value '{option_text}'.")
                            return True
                        except NoSuchElementException:
                            print(f"  Error: Option '{option_text}' not found by text or value in select element {element_id}")
                            return False
                else:
                    print(f"  Error: Element {element_id} not found or is not a select element for select action.")
                    return False
            elif action_type == "navigate":
                url = action.get('url')
                if url:
                    print(f"  Navigating to new URL: {url}")
                    self.driver.get(url)
                    WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body"))) # Wait for body
                    return True
                else:
                    print("  Error: Navigate action missing URL.")
                    return False
            elif action_type == "ANSWER":
                 print(f"  ANSWER action received: {action.get('content')}")
                 return True
            elif action_type == "stop":
                print(f"  Execution stopped as planned: {action.get('reason', 'No reason specified')}")
                return True
            elif action_type == "error": # This usually comes from planning module itself
                print(f"  Execution failed due to planning error: {action.get('message', 'Unknown error')}")
                return False
            else:
                print(f"  Error: Unknown action type '{action_type}'")
                return False
        except WebDriverException as e:
            print(f"Execution Module: Selenium error during execution of {action_type}: {e}")
            return False
        except Exception as e:
            print(f"Execution Module: Unexpected error during execution of {action_type}: {e}")
            return False

    def _find_element_for_action(self, element_id: Optional[str], state: Dict[str, Any], comment: Optional[str]=None) -> Optional[WebElement]:
        if not element_id:
            print("  Error: No element_id provided for action.")
            return None

        # If state is a combined state from multiple screenshots, we need to search all interactive elements
        interactive_elements = state.get('interactive_elements', [])
        target_element_info = next((el for el in interactive_elements if el.get('id') == element_id), None)
        
        if not target_element_info:
            print(f"  Warning: Element info for ID '{element_id}' not found in perceived state. Planner comment: '{comment}'")
            # Fallback: If planner provided a comment that might contain a text hint
            if comment:
                text_from_comment = comment
                print(f"  Attempting fallback using text from comment: '{text_from_comment}'")
                try:
                    element = self.driver.find_element(By.XPATH, f"//*[normalize-space()='{text_from_comment}' or contains(normalize-space(), '{text_from_comment}') or @aria-label='{text_from_comment}' or @title='{text_from_comment}']")
                    if element.is_displayed() and element.is_enabled():
                        print(f"    Located element heuristically using comment text '{text_from_comment}'.")
                        return element
                except NoSuchElementException:
                    print(f"    Fallback using comment text '{text_from_comment}' failed.")
            return None

        css_selector = target_element_info.get('css_selector')
        xpath = target_element_info.get('xpath')

        # Priority 1: CSS Selector from perception
        if css_selector and css_selector != "locator_unavailable":
            try:
                element = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, css_selector))
                )
                print(f"    Located element for ID '{element_id}' using CSS Selector: '{css_selector}'")
                return element
            except Exception as e:
                print(f"    CSS Selector '{css_selector}' for ID '{element_id}' not found or not interactable: {type(e).__name__}")

        # Priority 2: XPath from perception
        if xpath and xpath != "locator_unavailable":
            try:
                element = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, xpath))
                )
                print(f"    Located element for ID '{element_id}' using XPath: '{xpath}'")
                return element
            except Exception as e:
                print(f"    XPath '{xpath}' for ID '{element_id}' not found or not interactable: {type(e).__name__}")

        # Fallback: Heuristic Locators (if Perception failed to provide a working one)
        element_text = target_element_info.get('text')
        element_type = target_element_info.get('type', '').lower()
        visual_desc = target_element_info.get('visual_description', '')
        print(f"  Warning: Precise locator for ID '{element_id}' failed or unavailable. Falling back to heuristic search.")

        # Simplified heuristic: primarily by text, then by aria-label or title from visual_desc
        search_texts = [t for t in [element_text, visual_desc] if t] # Consider text and description

        for text_to_find in search_texts:
            if not text_to_find: continue
            try:
                # Try common attributes
                common_xpaths = [
                    f".//a[normalize-space()='{text_to_find}']",
                    f".//button[normalize-space()='{text_to_find}']",
                    f".//input[@value='{text_to_find}']",
                    f".//input[@aria-label='{text_to_find}']",
                    f".//button[@aria-label='{text_to_find}']",
                    f".//*[@title='{text_to_find}']",
                    f"//*[normalize-space()='{text_to_find}']", # General text match
                    f"//*[contains(normalize-space(), '{text_to_find}')]", # General contains match
                ]
                if element_type == 'input':
                    common_xpaths.append(f".//input[@placeholder='{text_to_find}']")

                for xp in common_xpaths:
                    try:
                        elements = self.driver.find_elements(By.XPATH, xp)
                        for el in elements:
                            if el.is_displayed() and el.is_enabled():
                                print(f"    Located element for ID '{element_id}' heuristically using XPath: {xp}")
                                return el
                    except NoSuchElementException:
                        continue
            except Exception as e_heuristic:
                print(f"    Error during heuristic search for '{text_to_find}': {e_heuristic}")
        
        print(f"  Error: Execution module could not locate element for ID '{element_id}' using any method.")
        return None


# Main Agent Class
class WebAgent:
    """Web automation agent that uses visual perception and reasoning to complete tasks"""
    def __init__(self, goal: str, start_url: str, batch_mode: bool = False, html_only_mode: bool = False, max_scrolls: int = 10):
        """
        Initialize WebAgent
        
        Args:
            goal: User's task instruction
            start_url: Starting website URL
            batch_mode: When True, screenshots are collected before processing together
            html_only_mode: When True, only HTML content is used (no screenshots)
            max_scrolls: Maximum number of page scrolls to capture content
        """
        self.goal = goal
        self.start_url = start_url
        self.batch_mode = batch_mode
        self.html_only_mode = html_only_mode
        self.max_scrolls = max_scrolls
        self.cookie_handled_domains = set()  # Track domains where cookie consent was already handled
        self.scroll_positions = []  # Track scroll positions for each step
        self.answer_scroll_position = 0  # The scroll position where answer was found
        self.screenshot_files = []  # Track screenshot files for cleanup

        os.makedirs("tmp", exist_ok=True)
        
        options = webdriver.ChromeOptions()
        # options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')

        try:
            self.driver = webdriver.Chrome(options=options)
        except WebDriverException as e:
            print(f"Error initializing WebDriver: {e}. Ensure ChromeDriver is in PATH or accessible.")
            exit()

        self.driver.maximize_window()

        self.perception_module = PerceptionModule()
        self.planning_module = PlanningReasoningModule()
        self.execution_module = ExecutionModule(self.driver)
        self.history = []
        self.max_steps = 15
        self.final_answer = None

    def is_page_bottom(self) -> bool:
        """Detects if scrolling has reached the bottom of the page"""
        current_scroll_y = self.driver.execute_script("return window.pageYOffset;")
        total_height = self.driver.execute_script("return document.body.scrollHeight;")
        viewport_height = self.driver.execute_script("return window.innerHeight;")
        
        return current_scroll_y + viewport_height + 10 >= total_height

    def scroll_page(self) -> bool:
        """Scrolls down by one viewport height and returns if position changed"""
        previous_position = self.driver.execute_script("return window.pageYOffset;")
        self.driver.execute_script("window.scrollBy(0, window.innerHeight);")
        time.sleep(1)
        current_position = self.driver.execute_script("return window.pageYOffset;")
        
        return previous_position != current_position

    def capture_full_page(self) -> List[Dict[str, Any]]:
        """Captures entire page through scrolling and screenshot analysis
        
        In batch mode, all screenshots are collected first, then processed together.
        Otherwise, each screenshot is processed immediately after capture.
        In HTML-only mode, page is still scrolled but no screenshots are taken.
        """
        print("\n--- Capturing Full Page Content ---")
        perception_results = []
        self.screenshot_files = []
        scroll_count = 0
        self.scroll_positions = []
        
        # Save initial scroll position (usually 0)
        current_position = self.driver.execute_script("return window.pageYOffset;")
        self.scroll_positions.append(current_position)
        
        # In HTML-only mode, we already scrolled in the run method, just return minimal state
        if self.html_only_mode:
            perception_results = [{
                "summary": "HTML-only mode - full page scrolled for content loading",
                "interactive_elements": [],
                "content_elements": [],
                "scroll_positions": self.scroll_positions
            }]
            return perception_results
        
        # Standard screenshot mode - similar to before but with scroll position tracking
        while scroll_count < self.max_scrolls:
            # Take screenshot of current viewport
            screenshot_file = f"tmp/screenshot_scroll_{scroll_count}.png"
            try:
                if not self.driver.save_screenshot(screenshot_file):
                    print(f"Error: Failed to save screenshot {screenshot_file}")
                    break
            except WebDriverException as e:
                print(f"Error saving screenshot: {e}")
                break
            
            # Store the screenshot filename
            self.screenshot_files.append(screenshot_file)
            print(f"Captured viewport {scroll_count+1} at position {self.scroll_positions[-1]}")
            
            # In non-batch mode, process each screenshot immediately
            if not self.batch_mode:
                current_perception = self.perception_module.perceive(screenshot_file)
                if "error" in current_perception:
                    print(f"Perception failed: {current_perception['error']}")
                    break
                # Add scroll position info to perception data
                current_perception["scroll_position"] = self.scroll_positions[-1]
                current_perception["scroll_index"] = scroll_count
                perception_results.append(current_perception)
                print(f"Processed viewport {scroll_count+1}")
            
            # Check if we're at the bottom
            if self.is_page_bottom():
                print("Reached the bottom of the page")
                break
                
            # Scroll down
            if not self.scroll_page():
                print("Scroll didn't change position, likely at bottom")
                break
                
            # Record new scroll position
            current_position = self.driver.execute_script("return window.pageYOffset;")
            self.scroll_positions.append(current_position)
            
            scroll_count += 1
        
                    # In batch mode, process all screenshots after completing scrolling
        if self.batch_mode:
            print(f"Batch processing {len(self.screenshot_files)} screenshots...")
            if self.screenshot_files:
                # Use the new batch perception method to process all screenshots at once
                batch_results = self.perception_module.perceive_batch(self.screenshot_files)
                if batch_results and "error" not in batch_results[0]:
                    perception_results = batch_results
                    # Add scroll position info to batch result
                    perception_results[0]["scroll_positions"] = self.scroll_positions
                    # Add scroll_index mapping to make it easier to reference specific viewport
                    perception_results[0]["scroll_indexes"] = list(range(len(self.scroll_positions)))
                    print(f"Successfully processed {len(self.screenshot_files)} screenshots in batch mode")
                else:
                    error_msg = batch_results[0].get("error", "Unknown batch processing error")
                    print(f"Batch perception failed: {error_msg}")
                    
                    # Fallback to individual processing if batch fails
                    print("Falling back to individual processing...")
                    for i, sf in enumerate(self.screenshot_files):
                        current_perception = self.perception_module.perceive(sf)
                        if "error" not in current_perception:
                            current_perception["scroll_position"] = self.scroll_positions[i] if i < len(self.scroll_positions) else 0
                            current_perception["scroll_index"] = i
                            perception_results.append(current_perception)
                            print(f"Processed screenshot {i+1}")
            else:
                print("No screenshots captured")
                
        # Perception modules handle API uploads deletion, but we keep track of local files
        print(f"Completed page capture with {len(self.screenshot_files)} viewports")
        
        # Scroll back to top for consistency
        self.driver.execute_script("window.scrollTo(0, 0);")
        
        return perception_results

    def handle_cookie_consent(self):
        """Tries to decline cookie collection on website if such prompt exists"""
        print("Attempting to decline cookie collection...")
        try:
            html_content = self.driver.page_source
            processed_html = self.planning_module._preprocess_html(html_content)
            
            # Query text model to identify how to decline cookies
            prompt = f"""
            Analyze the HTML of this webpage and help me decline any cookie collection prompts.
            If you find elements related to cookie consent, provide:
            1. A detailed description of the element to click to decline cookies or minimize data collection
            2. Any relevant CSS selector or XPath to locate the element
            
            Provide your response as JSON in this format:
            {{
                "found_cookie_prompt": true/false,
                "decline_button_description": "description of the button/element",
                "css_selector": "specific CSS selector if identifiable",
                "xpath": "specific XPath if identifiable"
            }}
            
            HTML Content:
            {processed_html[:50000]}
            """
            
            response = GEMINI_CLIENT.models.generate_content(model=TEXT_MODEL_NAME, contents=prompt)
            response_text = response.text.strip()
            if response_text.startswith("```json"): response_text = response_text[7:-3]
            elif response_text.startswith("```"): response_text = response_text[3:-3]
            
            try:
                cookie_info = json.loads(response_text)
                if cookie_info.get("found_cookie_prompt", False):
                    print(f"Cookie prompt detected: {cookie_info.get('decline_button_description')}")
                    
                    # Try CSS selector if available
                    if cookie_info.get("css_selector"):
                        try:
                            element = WebDriverWait(self.driver, 5).until(
                                EC.element_to_be_clickable((By.CSS_SELECTOR, cookie_info.get("css_selector")))
                            )
                            element.click()
                            print("Declined cookies using CSS selector")
                            return True
                        except Exception as e:
                            print(f"CSS selector method failed: {e}")
                    
                    # Try XPath if available
                    if cookie_info.get("xpath"):
                        try:
                            element = WebDriverWait(self.driver, 5).until(
                                EC.element_to_be_clickable((By.XPATH, cookie_info.get("xpath")))
                            )
                            element.click()
                            print("Declined cookies using XPath")
                            return True
                        except Exception as e:
                            print(f"XPath method failed: {e}")
                    
                    # Generic approach if selectors fail
                    common_cookie_button_texts = ["Decline", "Reject", "No, thanks", "Decline All", 
                                                 "I do not accept", "Necessary cookies only", "Continue without accepting"]
                    for button_text in common_cookie_button_texts:
                        try:
                            xpath = f"//*[contains(text(), '{button_text}') or contains(@aria-label, '{button_text}')]"
                            elements = self.driver.find_elements(By.XPATH, xpath)
                            for element in elements:
                                if element.is_displayed() and element.is_enabled():
                                    element.click()
                                    print(f"Declined cookies by clicking '{button_text}' button")
                                    return True
                        except Exception:
                            continue
                    
                    print("Could not automatically decline cookies despite detecting prompt")
                else:
                    print("No cookie consent prompt detected")
            except json.JSONDecodeError:
                print("Failed to parse model response about cookie consent")
                
        except Exception as e:
            print(f"Error while handling cookie consent: {e}")
        
        return False

    def _extract_domain(self, url):
        """Extract the main domain from a URL"""
        try:
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            # Extract main domain (e.g., 'example.com' from 'subdomain.example.com')
            parts = domain.split('.')
            if len(parts) > 2:
                # This handles cases like 'www.example.com'
                domain = '.'.join(parts[-2:])
            return domain
        except Exception as e:
            print(f"Error extracting domain from URL: {e}")
            return url  # Return full URL as fallback
    
    def run(self):
        """Main execution loop that processes user instructions"""
        operation_mode = "HTML-only mode" if self.html_only_mode else ("Batch mode" if self.batch_mode else "Standard mode")
        print(f"WebAgent starting. Goal: {self.goal} ({operation_mode}, Max scrolls: {self.max_scrolls})")
        try:
            # Initial navigation
            self.driver.get(self.start_url)
            print(f"Navigated to: {self.start_url}")
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(2)
            
            # Get current domain
            current_url = self.driver.current_url
            current_domain = self._extract_domain(current_url)
            
            # Only handle cookie consent if this domain hasn't been processed before
            if current_domain not in self.cookie_handled_domains:
                print(f"First visit to {current_domain} - Processing cookie consent dialogs...")
                self.handle_cookie_consent()
                self.cookie_handled_domains.add(current_domain)
            else:
                print(f"Cookie consent already handled for domain: {current_domain}")

            for step in range(self.max_steps):
                print(f"\n--- Step {step + 1}/{self.max_steps} ---")
                current_url = self.driver.current_url
                
                # In HTML-only mode, we scroll first to load all content
                if self.html_only_mode:
                    print("HTML-only mode: Scrolling through page to load content...")
                    scroll_count = 0
                    self.scroll_positions = []  # Reset scroll positions
                    
                    # Save initial scroll position (usually 0)
                    current_position = self.driver.execute_script("return window.pageYOffset;")
                    self.scroll_positions.append(current_position)
                    
                    # Scroll through the page to trigger lazy-loading
                    while scroll_count < self.max_scrolls:
                        # Check if we're at the bottom
                        if self.is_page_bottom():
                            print("Reached the bottom of the page")
                            break
                            
                        # Scroll down
                        if not self.scroll_page():
                            print("Scroll didn't change position, likely at bottom")
                            break
                            
                        # Record scroll position
                        current_position = self.driver.execute_script("return window.pageYOffset;")
                        self.scroll_positions.append(current_position)
                        
                        scroll_count += 1
                        print(f"Scrolled to position {current_position} (scroll {scroll_count})")
                    
                    # Scroll back to top for consistent starting point
                    self.driver.execute_script("window.scrollTo(0, 0);")
                    print("Scrolled back to top for analysis")
                
                # Get HTML document (always needed)
                html_content = self.driver.page_source
                
                # Handle HTML-only mode vs screenshot mode
                if self.html_only_mode:
                    print("Using HTML-only mode (no screenshots)")
                    
                    # Create a minimal perception state that satisfies the interface requirements
                    # but doesn't actually contain real screenshot data
                    screenshot_perceptions = [{
                        "summary": "HTML-only mode - full page scrolled for content loading",
                        "interactive_elements": [],
                        "content_elements": [],
                        "scroll_positions": self.scroll_positions
                    }]
                    
                    # Analyze with HTML content as the primary source
                    action = self.planning_module.analyze_html_only(
                        self.goal,
                        html_content,
                        current_url,
                        self.history
                    )
                else:
                    # Capture visual understanding of page using screenshots
                    screenshot_perceptions = self.capture_full_page()
                    if not screenshot_perceptions:
                        print("Error: Failed to capture any page content")
                        self.final_answer = "Error: Failed to capture page content"
                        break
                    
                    # Analyze and plan next action using combined data
                    action = self.planning_module.analyze_combined_data(
                        self.goal,
                        screenshot_perceptions,
                        html_content,
                        current_url,
                        self.history
                    )
                
                if not action or 'action_type' not in action:
                    print("Planning module returned invalid action. Stopping.")
                    self.final_answer = "Error: Planning module returned invalid action."
                    break

                # Use first screenshot for execution reference
                reference_state = screenshot_perceptions[0]
                
                # Record action in history
                page_summary = reference_state.get('summary', 'No summary')[:100]
                current_action_for_history = {'action': action, 'url': current_url, 'summary': page_summary}
                self.history.append(current_action_for_history)

                # Process answer or termination conditions
                if action.get('action_type') == 'ANSWER':
                    self.final_answer = action.get('content')
                    print(f"Goal achieved! Final Answer: {self.final_answer}")
                    
                    # Scroll to the position where the answer was found
                    if self.html_only_mode and 'scroll_position' in action:
                        try:
                            scroll_pos = action.get('scroll_position', 0)
                            print(f"Scrolling to position where answer was found: {scroll_pos}")
                            self.driver.execute_script(f"window.scrollTo(0, {scroll_pos});")
                            self.answer_scroll_position = scroll_pos
                        except Exception as e:
                            print(f"Failed to scroll to answer position: {e}")
                    elif 'scroll_index' in action:
                        try:
                            scroll_idx = action.get('scroll_index', 0)
                            # Check if we have enough scroll positions
                            if 0 <= scroll_idx < len(self.scroll_positions):
                                scroll_pos = self.scroll_positions[scroll_idx]
                                print(f"Scrolling to position where answer was found (index {scroll_idx}): {scroll_pos}")
                                self.driver.execute_script(f"window.scrollTo(0, {scroll_pos});")
                                self.answer_scroll_position = scroll_pos
                            else:
                                # In batch mode, we might have a single perception state with multiple scroll positions
                                # Try to get positions from reference_state
                                scroll_positions = reference_state.get('scroll_positions', [])
                                if 0 <= scroll_idx < len(scroll_positions):
                                    scroll_pos = scroll_positions[scroll_idx]
                                    print(f"Scrolling to position where answer was found (batch index {scroll_idx}): {scroll_pos}")
                                    self.driver.execute_script(f"window.scrollTo(0, {scroll_pos});")
                                    self.answer_scroll_position = scroll_pos
                                else:
                                    print(f"Invalid scroll index {scroll_idx}, not scrolling")
                        except Exception as e:
                            print(f"Failed to scroll to answer position: {e}")
                    
                    break
                
                if action.get('action_type') == 'stop':
                    reason = action.get('reason', 'No reason specified')
                    print(f"Planning module decided to stop: {reason}")
                    if not self.final_answer: 
                        self.final_answer = f"Stopped by planner: {reason}"
                    break
                    
                if action.get('action_type') == 'error':
                    message = action.get('message', 'Unknown planning error')
                    print(f"Planning failed: {message}. Stopping agent.")
                    self.final_answer = f"Planning Error: {message}"
                    break

                # Execute planned action
                success = self.execution_module.execute(action, reference_state)
                
                if not success:
                    print("Execution module reported failure for action. Stopping agent.")
                    self.final_answer = f"Execution failed for action: {json.dumps(action)}"
                    break
                
                # Wait for page updates after interactive actions
                if action.get("action_type") in ["click", "type", "select", "navigate"]:
                    print("Waiting for page to settle after action...")
                    time.sleep(3)
                    try:
                        WebDriverWait(self.driver, 7).until(
                           lambda d: d.execute_script('return document.readyState') == 'complete'
                        )
                        print("Page state complete.")
                    except TimeoutException:
                        print("Page did not reach 'complete' state within timeout after action.")
                    time.sleep(1)

            else:  # for loop finished without break (max_steps reached)
                print(f"\nMaximum steps ({self.max_steps}) reached. Stopping agent.")
                if not self.final_answer: 
                    self.final_answer = "Max steps reached without finding an answer."

            # scroll to the position where the answer was found
            if self.answer_scroll_position > 0:
                try:
                    print(f"Ensuring final scroll to position: {self.answer_scroll_position}")
                    self.driver.execute_script(f"window.scrollTo(0, {self.answer_scroll_position});")
                except Exception as e:
                    print(f"Final scroll failed: {e}")
                    
            # Clean up local screenshot files
            self._cleanup_screenshots()
            
            if self.final_answer:
                print(f"\n--- Final Result ---")
                print(f"Agent goal: {self.goal}")
                print(f"Agent answer/outcome: {self.final_answer}")
            else:
                print("\nAgent stopped without providing a final answer.")

        except WebDriverException as e:
            print(f"WebAgent encountered a WebDriver error: {e}")
            self.final_answer = f"WebDriver Error: {e}"
        except Exception as e:
            print(f"WebAgent encountered an unexpected error: {e}")
            import traceback
            traceback.print_exc()
            self.final_answer = f"Unexpected Error: {e}"
        finally:
            # Clean up screenshots if not done already
            self._cleanup_screenshots()
            print("WebAgent run finished.")

    def _cleanup_screenshots(self):
        """Remove all screenshot files after task completion"""
        if self.screenshot_files:
            print(f"Cleaning up {len(self.screenshot_files)} screenshot files...")
            for file_path in self.screenshot_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error removing file {file_path}: {e}")
            self.screenshot_files = []


if __name__ == "__main__":
    print("Welcome to WebAgent!")
    print("You can use commands like '-help', '-url', '-set' to configure the agent.")
    print("Type '-help' for detailed instructions.")

    # Default settings
    TARGET_URL = "https://www.imdb.com/"
    agent = None
    batch_mode = True  # Controls whether screenshots are processed in batch
    html_only_mode = False  # Controls whether to use only HTML or screenshots+HTML
    max_scrolls = 10  # Maximum number of screenshot scrolls
    
    def print_current_settings():
        """Display current agent settings"""
        mode = "HTML-only" if html_only_mode else ("Batch" if batch_mode else "Standard")
        print(f"\nCurrent Settings:")
        print(f"  Target URL: {TARGET_URL}")
        print(f"  Mode: {mode}")
        print(f"  Max Scrolls: {max_scrolls}")
    
    def display_help():
        """Display help information about available commands"""
        print("\n=== WebAgent Help ===")
        print("Available commands:")
        print("  -help                    Display this help message")
        print(f"  -url [new_url]           Change the target URL (default: {TARGET_URL})")
        print("  -set mode [option]       Set operation mode")
        print("                           Options: html, batch, standard")
        print("                             html: Only use HTML for analysis (much quicker without screenshots)")
        print("                             batch: Capture all screenshots before processing (default, more accurate)")
        print("                             standard: Process each screenshot immediately")
        print(f"  -set scrolls [number]    Set maximum number of page scrolls (default: {max_scrolls})")
        print("  quit                     Exit the program")
        print("\nExample usage:")
        print("  -set mode html           Switch to HTML-only mode")
        print("  -set scrolls 5           Set maximum scrolls to 5")
        print("  -url https://example.com Change target URL to example.com")
        print("\nTo run a task, simply type your instruction after the prompt.")
    
    def parse_set_command(command):
        """Parse -set command and update settings"""
        global batch_mode, html_only_mode, max_scrolls
        
        parts = command.split()
        if len(parts) < 3:
            print("Error: Invalid -set command format. Use: -set [parameter] [value]")
            return
        
        param = parts[1].lower()
        value = parts[2].lower()
        
        if param == "mode":
            if value == "html":
                html_only_mode = True
                batch_mode = False
                print("Mode set to HTML-only")
            elif value == "batch":
                html_only_mode = False
                batch_mode = True
                print("Mode set to Batch (screenshots processed together)")
            elif value == "standard":
                html_only_mode = False
                batch_mode = False
                print("Mode set to Standard (screenshots processed individually)")
            else:
                print(f"Error: Unknown mode '{value}'. Valid options: html, batch, standard")
        elif param == "scrolls":
            try:
                new_scrolls = int(value)
                if new_scrolls > 0:
                    max_scrolls = new_scrolls
                    print(f"Max scrolls set to {max_scrolls}")
                else:
                    print("Error: Max scrolls must be positive")
            except ValueError:
                print(f"Error: '{value}' is not a valid number for max scrolls")
        else:
            print(f"Error: Unknown parameter '{param}'. Valid parameters: mode, scrolls")
        
        if agent:
            agent.html_only_mode = html_only_mode
            agent.batch_mode = batch_mode
            agent.max_scrolls = max_scrolls
    
    while True:
        try:
            print_current_settings()
            user_input_goal = input("\nPlease enter your instruction (or type -help for commands): ").strip()
            
            if user_input_goal.lower() == 'quit':
                print("Exiting program...")
                break
            
            # Handle help command
            if user_input_goal.lower() == '-help':
                display_help()
                continue
            
            # Handle -url command
            if user_input_goal.lower().startswith('-url'):
                parts = user_input_goal.split(maxsplit=1)
                if len(parts) > 1:
                    TARGET_URL = parts[1]
                    print(f"Target URL updated to: {TARGET_URL}")
                else:
                    new_url = input(f"Enter new target URL (current: {TARGET_URL}): ").strip()
                    if new_url:
                        TARGET_URL = new_url
                continue
            
            # Handle -set command
            if user_input_goal.lower().startswith('-set'):
                parse_set_command(user_input_goal)
                continue

            if agent is None:
                print(f"Initializing new WebAgent instance...")
                agent = WebAgent(
                    goal=user_input_goal, 
                    start_url=TARGET_URL, 
                    batch_mode=batch_mode,
                    html_only_mode=html_only_mode,
                    max_scrolls=max_scrolls
                )
            else:
                # Reuse existing agent and driver if possible, reset state
                print(f"Reusing WebAgent instance. New Goal: {user_input_goal}")
                agent.goal = user_input_goal
                agent.history = []
                agent.final_answer = None
                agent.html_only_mode = html_only_mode
                agent.batch_mode = batch_mode
                agent.max_scrolls = max_scrolls
                
                # Navigate to start_url again if it's different or for a fresh start
                if agent.driver.current_url != TARGET_URL:
                    print(f"Navigating to base URL for new task: {TARGET_URL}")
                    agent.start_url = TARGET_URL
                    agent.driver.get(TARGET_URL)
                    WebDriverWait(agent.driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                else:
                    print(f"Agent already at {agent.driver.current_url}, proceeding with new goal.")

            agent.run()
            
        except KeyboardInterrupt:
            print("\nCtrl+C detected, exiting program...")
            break
        except Exception as e:
            print(f"An error occurred in the main loop: {e}")
            import traceback
            traceback.print_exc()
            if agent and hasattr(agent, 'driver') and agent.driver:
                agent.driver.quit()
                agent = None
            print("Agent reset due to error. Please try again.")
        
    if agent and hasattr(agent, 'driver') and agent.driver:
        print("Quitting WebDriver...")
        agent.driver.quit()
    print("Program terminated.")
