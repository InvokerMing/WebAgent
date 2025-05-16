from google import genai
import json
import time
from pathlib import Path

from bs4 import BeautifulSoup
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.common.exceptions import NoSuchElementException, WebDriverException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from typing import List, Dict, Any, Optional
from selenium.webdriver.support.ui import Select

# --- Configuration ---
GEMINI_API_KEY = "APIKEY"
VISION_MODEL_NAME = "gemini-2.0-flash"
TEXT_MODEL_NAME = "gemini-2.0-flash"
TARGET_URL = "https://movie.douban.com/tv/"

try:
    GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"Error configuring Gemini API: {e}. Please ensure GEMINI_API_KEY is set correctly.")
    exit()

# --- 1. Perception Module ---
class PerceptionModule:
    """
    Responsible for perceiving the web environment, processing screenshots and optional HTML information.
    """
    def __init__(self):
        self.model = GEMINI_CLIENT.models

    def perceive(self, screenshot_path: str, simplified_html: Optional[str] = None) -> Dict[str, Any]:
        print(f"Perception Module: Analyzing screenshot {screenshot_path}...")
        if not Path(screenshot_path).is_file():
            return {"error": f"Screenshot file not found: {screenshot_path}"}

        prompt_text = f"""
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


# --- 2. Planning & Reasoning Module ---
class PlanningReasoningModule:
    def __init__(self):
        self.model = GEMINI_CLIENT.models

    def _preprocess_html(self, html_content: str, max_length: int = 1000000) -> str:
        """
        Preprocess HTML content to make it more concise while preserving important information.
        """
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


    def plan_action(self, goal: str, current_state: Dict[str, Any], history: List[Dict[str, Any]],
                    current_url: str, is_page_bottom: bool,
                    html_content: Optional[str] = None, use_dual_mode: bool = False) -> Dict[str, Any]:
        print("\n=== Planning Module: Starting Action Planning ===")
        if "error" in current_state:
            return {'action_type': 'error', 'message': f'Perception module failed: {current_state["error"]}'}

        print("\n--- Vision-based Planning ---")
        vision_plan = self._get_vision_plan(goal, current_state, history, current_url, is_page_bottom)
        print(f"Vision model's plan: {json.dumps(vision_plan, indent=2, ensure_ascii=False)}")

        if use_dual_mode and html_content:
            print("\n--- HTML-based Planning ---")
            # Pass current_state to plan_action_from_html so it knows about visually identified elements
            html_plan = self.plan_action_from_html(goal, html_content, current_state, history, current_url, is_page_bottom)
            print(f"Text model's plan: {json.dumps(html_plan, indent=2, ensure_ascii=False)}")

            print("\n--- Plan Comparison ---")
            final_plan = self.compare_plans(vision_plan, html_plan, goal, current_state) # Pass current_state for context
            print(f"Final chosen plan: {json.dumps(final_plan, indent=2, ensure_ascii=False)}")
        else:
            final_plan = vision_plan

        print("\n=== Planning Module: Action Planning Completed ===\n")
        return final_plan

    def _get_vision_plan(self, goal: str, current_state: Dict[str, Any], history: List[Dict[str, Any]],
                         current_url: str, is_page_bottom: bool) -> Dict[str, Any]:
        print("Generating plan based on visual analysis...")
        prompt = f"""
        You are a web assistant. Decide the next best action based on the user's goal, current web page state, action history, current URL, and scroll state.

        User Goal: {goal}
        Current URL: {current_url}
        Is Page Bottom Reached: {is_page_bottom}

        Current Web Page State Summary: {current_state.get('summary', 'No summary available')}

        Interactive Elements on Page (use their 'id' field to specify them in your action):
        {self._format_elements(current_state.get('interactive_elements', []))}

        Relevant Content Elements on Page:
        {self._format_content(current_state.get('content_elements', []))}

        Action History (last {len(history)} steps, check for loops or stagnation):
        {self._format_history(history)}

        Based on the goal, current page, and history, decide the next step.
        If the goal seems achievable with current information, use "ANSWER".
        If more content might be below and `is_page_bottom` is false, consider "scroll".
        If stuck or goal unachievable, use "stop".

        Your response MUST be one of the following JSON formats:
        1. Click element: {{"action_type": "click", "element_id": "element_id_to_click", "comment": "Reason for clicking this element"}}
        2. Type text: {{"action_type": "type", "element_id": "input_element_id", "text": "text_to_type", "comment": "Reason for typing this"}}
        3. Select option: {{"action_type": "select", "element_id": "select_element_id", "option_text": "text_of_option_to_select", "comment": "Reason for selecting"}}
        4. Scroll page: {{"action_type": "scroll", "direction": "down_one_viewport"}} (Use this if more content is needed and not at bottom)
           (No "up" scroll needed for now, focus on progressive discovery. "down" is the only scroll direction.)
        5. Answer the goal: {{"action_type": "ANSWER", "content": "The answer based on current content elements"}}
        6. Stop/Finished: {{"action_type": "stop", "reason": "Explain why (e.g., goal achieved, task seems impossible, stuck in a loop)"}}
        7. Navigate: {{"action_type": "navigate", "url": "target_url", "comment": "Reason for navigating, use sparingly"}}

        Output ONLY the JSON for the next action. Ensure your reasoning is sound.
        If an element from 'Interactive Elements' has "locator_unavailable" for both css_selector and xpath, avoid choosing it for click/type/select unless absolutely no other option and you describe it by text.
        """
        try:
            response = self.model.generate_content(model=TEXT_MODEL_NAME, contents=prompt)
            response_text = response.text.strip()
            if response_text.startswith("```json"): response_text = response_text[7:-3]
            elif response_text.startswith("```"): response_text = response_text[3:-3]
            return json.loads(response_text)
        except Exception as e:
            print(f"Vision-based planning failed: {e}")
            return {'action_type': 'stop', 'reason': f'Vision planning error: {e}'}


    def plan_action_from_html(self, goal: str, html_content: str, current_state: Dict[str, Any],
                              history: List[Dict[str, Any]], current_url: str, is_page_bottom: bool) -> Dict[str, Any]:
        print("Generating plan based on HTML analysis...")
        processed_html = self._preprocess_html(html_content)
        print(f"HTML content length after preprocessing for planning: {len(processed_html)} characters")

        prompt = f"""
        You are a web assistant. Analyze the provided HTML content and decide the next best action based on the user's goal.
        You are also given a list of interactive elements identified from a *visual screenshot* of the current page.
        Try to map your findings from the HTML to these visually identified elements if it helps in choosing a target.
        If you choose an action on an element, refer to it using its 'id' from the 'Interactive Elements from Visual Analysis' list.

        User Goal: {goal}
        Current URL: {current_url}
        Is Page Bottom Reached: {is_page_bottom}

        Interactive Elements from Visual Analysis (use their 'id' field if targeting one of them):
        {self._format_elements(current_state.get('interactive_elements', []))}

        Processed HTML Content of the current page:
        {processed_html}

        Action History (last {len(history)} steps, check for loops or stagnation):
        {self._format_history(history)}

        Based on the HTML, visual elements, goal, and history, decide the next step.
        If the goal seems achievable with current information, use "ANSWER".
        If more content might be below (check HTML structure) and `is_page_bottom` is false, consider "scroll".
        If stuck or goal unachievable, use "stop".

        Your response MUST be one of the following JSON formats, using the 'element_id' from the Visual Analysis list if targeting one of those elements:
        1. Click element: {{"action_type": "click", "element_id": "element_id_from_visual_list", "comment": "Reason for clicking this element based on HTML and visual context"}}
        2. Type text: {{"action_type": "type", "element_id": "input_element_id_from_visual_list", "text": "text_to_type", "comment": "Reason for typing this"}}
        3. Select option: {{"action_type": "select", "element_id": "select_element_id_from_visual_list", "option_text": "text_of_option_to_select", "comment": "Reason for selecting"}}
        4. Scroll page: {{"action_type": "scroll", "direction": "down_one_viewport"}}
        5. Answer the goal: {{"action_type": "ANSWER", "content": "The answer based on HTML content and visual elements"}}
        6. Stop/Finished: {{"action_type": "stop", "reason": "Explain why (e.g., goal achieved from HTML, task impossible, stuck)"}}
        7. Navigate: {{"action_type": "navigate", "url": "target_url", "comment": "Reason for navigating, use sparingly based on HTML findings"}}

        Output ONLY the JSON for the next action. Ensure your reasoning is sound.
        If an element from 'Interactive Elements' has "locator_unavailable" for both css_selector and xpath, avoid choosing it for click/type/select unless absolutely no other option and you describe it by text from HTML.
        """
        try:
            response = self.model.generate_content(model=TEXT_MODEL_NAME, contents=prompt)
            response_text = response.text.strip()
            if response_text.startswith("```json"): response_text = response_text[7:-3]
            elif response_text.startswith("```"): response_text = response_text[3:-3]
            return json.loads(response_text)
        except Exception as e:
            print(f"HTML-based planning failed: {e}")
            return {'action_type': 'stop', 'reason': f'HTML planning error: {e}'}

    def compare_plans(self, vision_plan: Dict[str, Any], html_plan: Dict[str, Any], goal: str, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compares vision-based and HTML-based plans and decides which one to use.
        The core idea is to see if one plan is "safer" or more likely to succeed.
        """
        print("Comparing plans from both models...")

        # Priority: Error, Stop, Answer
        if vision_plan.get("action_type") == "error": return html_plan
        if html_plan.get("action_type") == "error": return vision_plan
        if vision_plan.get("action_type") == "stop": return vision_plan # If vision says stop, probably good reason
        if html_plan.get("action_type") == "stop" and vision_plan.get("action_type") != "ANSWER": return html_plan
        if vision_plan.get("action_type") == "ANSWER": return vision_plan # Vision answer is high confidence
        if html_plan.get("action_type") == "ANSWER": return html_plan

        # Heuristic: Prefer plans that target elements with good locators
        def get_element_locator_quality(plan: Dict[str, Any], state: Dict[str, Any]) -> int:
            element_id = plan.get("element_id")
            if not element_id or plan.get("action_type") not in ["click", "type", "select"]:
                return 0 # Not an element action or no element_id
            
            el_info = next((el for el in state.get('interactive_elements', []) if el.get('id') == element_id), None)
            if not el_info: return -1 # Element ID not found in visual perception

            has_css = el_info.get("css_selector") and el_info.get("css_selector") != "locator_unavailable"
            has_xpath = el_info.get("xpath") and el_info.get("xpath") != "locator_unavailable"

            if has_css or has_xpath: return 2 # Good locator
            return 1 # Element ID exists but no good locator from perception

        vision_quality = get_element_locator_quality(vision_plan, current_state)
        html_quality = get_element_locator_quality(html_plan, current_state)

        # Prefer plan with better locator quality for its target element
        if vision_quality > html_quality:
            print("Plan comparison: Vision plan chosen due to better target element locator quality or being a non-element action.")
            return vision_plan
        if html_quality > vision_quality:
            print("Plan comparison: HTML plan chosen due to better target element locator quality.")
            return html_plan
        
        # If quality is same, or both are non-element actions (like scroll)
        # Fallback to a simple preference or more advanced LLM comparison if needed.
        # For now, let's default to vision plan if qualities are equal and not -1.
        if vision_quality == -1 and html_quality == -1: # Both referred to non-existent element_ids
             print("Plan comparison: Both plans referred to non-existent element_ids. Defaulting to vision plan (which might be 'stop').")
             return vision_plan # Or could be a stop action if both are bad.
        
        print("Plan comparison: Locator qualities similar or non-element actions. Defaulting to vision plan.")
        return vision_plan # Default to vision if no strong reason for HTML
        
        # The LLM-based comparison can be a fallback if heuristics are not enough:
        # prompt = f""" ... (original compare_plans prompt) ... """
        # try: ... result = json.loads(response_text); return result['chosen_plan'] ...
        # except: return vision_plan

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


# --- 3. Execution Module ---

class ExecutionModule:
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
            # ... (type, select methods also use _find_element_for_action) ...
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
            elif action_type == "scroll":
                # direction = action.get('direction', 'down') # Original
                direction = action.get('direction', 'down_one_viewport') # From new planner prompt
                if direction == "down_one_viewport":
                    print(f"  Scrolling page down by one viewport")
                    self.driver.execute_script("window.scrollBy(0, window.innerHeight);")
                # Add other scroll types if planner supports them e.g. "to_element", "to_bottom"
                # else if direction == "up_one_viewport":
                #    self.driver.execute_script("window.scrollBy(0, -window.innerHeight);")
                else:
                    print(f"  Unknown scroll direction: {direction}")
                    return False
                return True
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

        target_element_info = next((el for el in state.get('interactive_elements', []) if el.get('id') == element_id), None)

        if not target_element_info:
            print(f"  Warning: Element info for ID '{element_id}' not found in perceived state. Planner comment: '{comment}'")
            # Fallback: If planner provided a comment that might contain a text hint
            if comment:
                # This is a weak fallback, relying on the planner's comment for text.
                # A better planner would put descriptive text in the action itself if element_id is unreliable.
                text_from_comment = comment # Simplistic, might need to parse comment
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
                # element = self.driver.find_element(By.CSS_SELECTOR, css_selector)
                # if element.is_displayed() and element.is_enabled(): # element_to_be_clickable checks this
                print(f"    Located element for ID '{element_id}' using CSS Selector: '{css_selector}'")
                return element
            except Exception as e: # TimeoutException or NoSuchElementException
                print(f"    CSS Selector '{css_selector}' for ID '{element_id}' not found or not interactable: {type(e).__name__}")

        # Priority 2: XPath from perception
        if xpath and xpath != "locator_unavailable":
            try:
                element = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, xpath))
                )
                # element = self.driver.find_element(By.XPATH, xpath)
                # if element.is_displayed() and element.is_enabled():
                print(f"    Located element for ID '{element_id}' using XPath: '{xpath}'")
                return element
            except Exception as e:
                print(f"    XPath '{xpath}' for ID '{element_id}' not found or not interactable: {type(e).__name__}")

        # Fallback: Heuristic Locators (if Perception failed to provide a working one)
        element_text = target_element_info.get('text')
        element_type = target_element_info.get('type', '').lower()
        visual_desc = target_element_info.get('visual_description', '')
        print(f"  Warning: Precise locator for ID '{element_id}' (CSS: '{css_selector}', XPath: '{xpath}') failed or unavailable. Falling back to heuristic search based on text='{element_text}', type='{element_type}', desc='{visual_desc}'.")

        # Simplified heuristic: primarily by text, then by aria-label or title from visual_desc
        # This part can be expanded significantly as in your original _find_element_for_action
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


# --- Main Agent Class ---
class SimpleWebAgent:
    def __init__(self, goal: str, start_url: str, use_dual_mode: bool = False):
        self.goal = goal
        self.start_url = start_url
        self.use_dual_mode = use_dual_mode

        options = webdriver.ChromeOptions()
        # options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        # options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

        try:
            self.driver = webdriver.Chrome(options=options)
        except WebDriverException as e:
            print(f"Error initializing WebDriver: {e}. Ensure ChromeDriver is in PATH or accessible.")
            exit()

        self.driver.set_window_size(1280, 900)

        self.perception_module = PerceptionModule()
        self.planning_module = PlanningReasoningModule()
        self.execution_module = ExecutionModule(self.driver)
        self.history = []
        self.max_steps = 15 # Can be adjusted
        self.final_answer = None
        # self.last_scroll_position = 0 # Removed

    def is_page_bottom(self) -> bool:
        """Check if we've scrolled to the effective bottom of the page."""
        # ScrollHeight might change as new content loads, so this is an approximation
        current_scroll_y = self.driver.execute_script("return window.pageYOffset;")
        total_height = self.driver.execute_script("return document.body.scrollHeight;")
        viewport_height = self.driver.execute_script("return window.innerHeight;")
        
        # Consider "bottom" if we are within a small tolerance, or if scrollY hasn't changed after a scroll attempt
        # For simplicity now, just check if we are near the end
        return current_scroll_y + viewport_height + 10 >= total_height # Added a small tolerance of 10px


    # scroll_page method is not needed here anymore as scrolling is an explicit action

    def run(self):
        print(f"WebAgent starting. Goal: {self.goal}")
        try:
            self.driver.get(self.start_url)
            print(f"Navigated to: {self.start_url}")
            # Wait for the body to be present after initial navigation
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(2) # Extra brief wait for initial scripts to settle

            for step in range(self.max_steps):
                print(f"\n--- Step {step + 1}/{self.max_steps} ---")
                current_url = self.driver.current_url
                page_is_bottom = self.is_page_bottom()
                print(f"Current URL: {current_url}, Page at bottom: {page_is_bottom}")


                # 1. Perceive
                screenshot_file = f"screenshot_step_{step + 1}.png"
                try:
                    if not self.driver.save_screenshot(screenshot_file):
                        print("Error: Failed to save screenshot.")
                        self.final_answer = "Error: Failed to save screenshot."
                        break
                except WebDriverException as e:
                    print(f"Error saving screenshot: {e}")
                    self.final_answer = f"Error: Failed to save screenshot - {e}"
                    break
                
                html_for_perception = None
                # if self.use_dual_mode: # Simplified HTML for perception module is optional
                #    html_for_perception = self.planning_module._preprocess_html(self.driver.page_source, max_length=20000)

                current_state = self.perception_module.perceive(screenshot_file, simplified_html=html_for_perception)

                if "error" in current_state:
                    print(f"Perception failed: {current_state['error']}. Stopping agent.")
                    self.final_answer = f"Perception Error: {current_state['error']}"
                    break
                
                page_summary_for_history = current_state.get('summary', 'No summary')[:100] # Truncate for history

                # 2. Plan
                full_html_for_planning = None
                if self.use_dual_mode:
                    full_html_for_planning = self.driver.page_source # Pass full source to HTML planner

                action = self.planning_module.plan_action(
                    self.goal,
                    current_state,
                    self.history,
                    current_url,
                    page_is_bottom,
                    html_content=full_html_for_planning,
                    use_dual_mode=self.use_dual_mode
                )

                if not action or 'action_type' not in action:
                    print("Planning module returned invalid action. Stopping.")
                    self.final_answer = "Error: Planning module returned invalid action."
                    break

                current_action_for_history = {'action': action, 'url': current_url, 'summary': page_summary_for_history}
                # Add to history *before* execution, so if execution fails, we know what was attempted
                self.history.append(current_action_for_history)


                if action.get('action_type') == 'ANSWER':
                    self.final_answer = action.get('content')
                    print(f"Goal achieved by Planner! Final Answer: {self.final_answer}")
                    break
                if action.get('action_type') == 'stop':
                    reason = action.get('reason', 'No reason specified')
                    print(f"Planning module decided to stop: {reason}")
                    if not self.final_answer: self.final_answer = f"Stopped by planner: {reason}"
                    break
                if action.get('action_type') == 'error': # Error from planner itself
                    message = action.get('message', 'Unknown planning error')
                    print(f"Planning failed: {message}. Stopping agent.")
                    self.final_answer = f"Planning Error: {message}"
                    break

                # 3. Execute
                success = self.execution_module.execute(action, current_state)

                if not success:
                    print("Execution module reported failure for action. Stopping agent.")
                    self.final_answer = f"Execution failed for action: {json.dumps(action)}"
                    # Potentially add a retry mechanism here or allow planner to react to failed execution
                    break
                
                # Wait for page to potentially update after action
                # This is a general wait. More specific waits (e.g., for an element to appear/disappear) would be better.
                if action.get("action_type") in ["click", "type", "select", "navigate"]:
                    print("Waiting for page to settle after action...")
                    time.sleep(3) # Shorter, but still a general wait
                    try:
                        WebDriverWait(self.driver, 7).until(
                           lambda d: d.execute_script('return document.readyState') == 'complete'
                        )
                        print("Page state complete.")
                    except TimeoutException:
                        print("Page did not reach 'complete' state within timeout after action.")
                    time.sleep(1) # Small additional buffer

            else: # for loop finished without break (max_steps reached)
                print(f"\nMaximum steps ({self.max_steps}) reached. Stopping agent.")
                if not self.final_answer: self.final_answer = "Max steps reached."

            if self.final_answer:
                print(f"\n--- Final Result ---")
                print(f"Agent goal: {self.goal}")
                print(f"Agent answer/outcome: {self.final_answer}")
            else:
                print("\nAgent stopped without providing a final answer (or max steps reached without answer).")

        except WebDriverException as e:
            print(f"WebAgent encountered a WebDriver error: {e}")
            self.final_answer = f"WebDriver Error: {e}"
        except Exception as e:
            print(f"WebAgent encountered an unexpected error: {e}")
            import traceback
            traceback.print_exc()
            self.final_answer = f"Unexpected Error: {e}"
        finally:
            print("WebAgent run finished. Cleaning up...")
            if hasattr(self, 'driver') and self.driver:
                # self.driver.quit() # Moved to main loop for persistence
                pass # Driver quit is handled in the __main__ loop now for iterative use
            
            # Clean up screenshots
            for i in range(self.max_steps + 2): # Iterate a bit beyond max_steps just in case
                sf = Path(f"screenshot_step_{i}.png")
                if sf.exists():
                    try:
                        sf.unlink()
                    except OSError as e_del:
                        print(f"Error deleting screenshot {sf}: {e_del}")
            print("Screenshots cleaned up.")


if __name__ == "__main__":
    print("Welcome to WebAgent!")

    TARGET_URL = "https://www.imdb.com/"

    agent = None
    
    use_dual_mode = True
    
    while True:
        try:
            print(f"\nCurrent Target URL for new tasks: {TARGET_URL}")
            user_input_goal = input("Please enter your instruction (or type 'quit' to exit, 'url' to change target URL): ").strip()
            
            if user_input_goal.lower() == 'quit':
                print("Exiting program...")
                break
            
            if user_input_goal.lower() == 'url':
                new_url = input(f"Enter new target URL (current: {TARGET_URL}): ").strip()
                if new_url:
                    TARGET_URL = new_url
                continue

            if agent is None:
                print(f"Initializing new WebAgent instance for URL: {TARGET_URL}")
                agent = SimpleWebAgent(goal=user_input_goal, start_url=TARGET_URL, use_dual_mode=use_dual_mode)
            else:
                # Reuse existing agent and driver if possible, reset state
                print(f"Reusing WebAgent instance. New Goal: {user_input_goal}")
                agent.goal = user_input_goal
                agent.history = []
                agent.final_answer = None
                # Optionally, navigate to start_url again if it's different or for a fresh start
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
