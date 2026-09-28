from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto("https://google.com")


    # Define a JavaScript function to traverse the DOM from a starting point
    def traverse_dom(selector):
        return page.evaluate("""
            (sel) => {
                const startNode = document.querySelector(sel);
                if (!startNode) return null;

                // Traversal example: Move to the parent, then to its next sibling
                const parent = startNode.parentElement;
                const sibling = parent ? parent.nextElementSibling : null;

                return sibling ? sibling.innerText : "No sibling found";
            }
        """, selector)


    result = traverse_dom("#my-element")
    print(f"Traversed element text: {result}")

    browser.close()
