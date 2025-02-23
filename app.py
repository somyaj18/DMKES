import re
import tkinter as tk
from tkinter import PhotoImage, filedialog
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import matplotlib as plt
import numpy as np
import nltk
import matplotlib.pyplot as plt
import openai
import PyPDF2
import fitz
import os
from PIL import Image
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
# Set OpenAI API key
openai.api_key = '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'


def calculate_matching_percentage():
    left_file_paths = left_entry.get().split('\n')
    right_file_paths = right_entry.get().split('\n')

    matching_percentages = []
    for left_file in left_file_paths:
        for right_file in right_file_paths:
            if left_file.strip() and right_file.strip():
                try:
                    left_text = read_pdf(left_file)
                    right_text = read_pdf(right_file)

                    left_keywords = extract_keywords(left_text)
                    right_keywords = extract_keywords(right_text)

                    matching_percentage = calculate_match_percentage(left_keywords, right_keywords)
                    matching_percentages.append(matching_percentage)
                except Exception as e:
                    messagebox.showerror("Error", f"Error processing files: {e}")
                    return

    if matching_percentages:
        result_label.config(text=f"Matching Percentages: {matching_percentages}%", fg="black")
        generate_documents_graph(matching_percentages)


def read_pdf(file_path):
    try:
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        raise ValueError(f"Error reading PDF: {e}")


def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]


def calculate_match_percentage(keywords1, keywords2):
    common_keywords = set(keywords1) & set(keywords2)
    total_keywords = set(keywords1) | set(keywords2)
    return round((len(common_keywords) / len(total_keywords)) * 100, 2) if total_keywords else 0


def browse_file(entry_widget):
    filenames = filedialog.askopenfilenames(filetypes=[("PDF files", "*.pdf")])
    if filenames:
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, "\n".join(filenames))


def generate_documents_graph(matching_percentages):
    # Prepare documents list for plotting
    documents = [f"Document {i + 1}" for i in range(len(matching_percentages))]

    plt.figure(figsize=(12, 6))

    # Bar chart
    plt.subplot(1, 3, 1)
    plt.bar(documents, matching_percentages, color='green')
    plt.xlabel('Documents')
    plt.ylabel('Match Percentage')
    plt.title('Document Matching Percentage')

    # Median chart
    median_percentage = np.median(matching_percentages)
    median_values = [median_percentage] * len(matching_percentages)

    plt.subplot(1, 3, 2)
    plt.plot(documents, median_values, color='red', linestyle='--')
    plt.scatter(documents, matching_percentages, color='blue')
    plt.xlabel('Documents')
    plt.ylabel('Match Percentage')
    plt.title('Median Chart')

    # Distribution curve
    plt.subplot(1, 3, 3)
    density = np.exp(-((np.array(matching_percentages) - np.mean(matching_percentages)) * 2) / (
                2 * np.std(matching_percentages) * 2))
    density /= np.max(density)  # normalize the density to fit in the plot
    plt.plot(documents, density, color='orange')
    plt.xlabel('Documents')
    plt.ylabel('Density')
    plt.title('Distribution Curve')

    plt.tight_layout()
    plt.show()


def save_highly_repeated_keywords(all_keywords, right_text, doc_path):
    # Set a threshold for keyword repetition
    repetition_threshold = 5

    # Count the occurrences of each keyword
    keyword_counts = {keyword: all_keywords.count(keyword) for keyword in set(all_keywords)}

    # Filter keywords that are repeated more than the threshold
    highly_repeated_keywords = [keyword for keyword, count in keyword_counts.items() if count > repetition_threshold]

    # Save and print highly repeated keywords to a file
    with open("highly_repeated_keywords.txt", "w", encoding='utf-8') as file:
        file.write("Highly Repeated Keywords:\n")
        file.write(", ".join(highly_repeated_keywords))

    # Print highly repeated keywords to the console
    print("Highly Repeated Keywords for", doc_path, ":", ", ".join(highly_repeated_keywords))

    # Open the document and create an annotation layer
    doc = fitz.open(doc_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        highlight = page.add_highlight_annot

        # Loop through each word and highlight it on the page if it's highly repeated
        for word in highly_repeated_keywords:
            for match in re.finditer(r'\b%s\b' % re.escape(word), right_text, re.IGNORECASE):
                # Get the coordinates of the word
                start_idx = match.start()
                end_idx = match.end()
                rect = page.search_for(right_text[start_idx:end_idx])

                # Highlight the word with a yellow rectangle
                highlight(rect)

    # Save the modified document with highlighted words
    modified_doc_path = doc_path.replace('.pdf', '_highlighted.pdf')
    doc.save(modified_doc_path, encryption=None, garbage=4, deflate=True, clean=True, encryption_filename=None,
             permissions=None, owner_pw=None, user_pw=None, use_aes=False, use_128bit=True, use_iso32000=False,
             compress_level=2, ascii=False, stream=False, encrypt=False, metadata=None, encoding='utf-8')
    doc.close()

    print(f"Highlighted PDF saved as: {modified_doc_path}")


def read_document_content(doc_path):
    _, file_extension = os.path.splitext(doc_path.lower())

    if file_extension == '.pdf':
        return read_pdf(doc_path)
    elif file_extension == '.txt':
        with open(doc_path, 'r', encoding='utf-8') as file:
            return file.read()

    return ""


def browse_file(entry_widget):
    filenames = filedialog.askopenfilenames()
    if filenames:
        # If files are selected, insert them into the entry widget
        entry_widget.delete(0, tk.END)
        for file in filenames:
            entry_widget.insert(tk.END, file + '\n')


def login(username_entry, password_entry):
    # Placeholder function for login logic
    username = username_entry.get()
    password = password_entry.get()
    print(f"Logging in with username: {username} and password: {password}")


def signup(username_entry, password_entry):
    # Placeholder function for signup logic
    username = username_entry.get()
    password = password_entry.get()
    # Call function to save user credentials
    save_user_credentials(username, password)
    print(f"Signing up with username: {username} and password: {password}")


def save_user_credentials(username, password):
    # Placeholder function for saving user credentials
    print(f"Saving username: {username} and password: {password}")


def open_login_window():
    login_window = tk.Toplevel(root)
    login_window.title("Login")
    login_window.geometry("800x600")  # Set the initial size of the login window
    login_window.configure(bg="#FFDAB9")  # Set background color to peach

    # Create and layout widgets for login functionality
    login_label = tk.Label(login_window, text="Login", font=("Arial", 18), bg="#FFDAB9")
    login_label.pack(pady=10)

    # Add Logo
    logo_img = PhotoImage(file="vector.png")  # Ensure correct path to the image file
    # Resize the logo image
    logo_img = logo_img.subsample(2, 2)  # Change subsample values to adjust the size
    logo_label = tk.Label(login_window, image=logo_img, bg="#FFDAB9")
    logo_label.image = logo_img  # Keep a reference to prevent garbage collection
    logo_label.pack(pady=10)

    frame = tk.Frame(login_window, bg="#FFDAB9")  # Frame for centering widgets
    frame.pack(expand=True)

    username_label = tk.Label(frame, text="Username:", font=("Arial", 12), bg="#FFDAB9")
    username_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
    username_entry = tk.Entry(frame, font=("Arial", 12))
    username_entry.grid(row=0, column=1, padx=5, pady=5)

    password_label = tk.Label(frame, text="Password:", font=("Arial", 12), bg="#FFDAB9")
    password_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
    password_entry = tk.Entry(frame, show="*", font=("Arial", 12))
    password_entry.grid(row=1, column=1, padx=5, pady=5)

    login_button = tk.Button(frame, text="Login", font=("Arial", 12),
                             command=lambda: login(username_entry, password_entry))
    login_button.grid(row=2, column=0, columnspan=2, padx=5, pady=10)


    # Center the frame
    frame.grid_columnconfigure(0, weight=1)
    frame.grid_columnconfigure(2, weight=1)
    frame.grid_rowconfigure(4, weight=1)


# Define function for opening signup window
def open_signup_window():
    signup_window = tk.Toplevel(root)
    signup_window.title("Sign Up")
    signup_window.geometry("800x600")  # Set the initial size of the signup window
    signup_window.configure(bg="#FFDAB9")  # Set background color to peach

    # Create and layout widgets for signup functionality
    signup_label = tk.Label(signup_window, text="Sign Up", font=("Arial", 18), bg="#FFDAB9")
    signup_label.pack(pady=10)

    # Add Logo
    logo_img = PhotoImage(file="vector.png")  # Ensure correct path to the image file
    # Resize the logo image
    logo_img = logo_img.subsample(2, 2)  # Change subsample values to adjust the size
    logo_label = tk.Label(signup_window, image=logo_img, bg="#FFDAB9")
    logo_label.image = logo_img  # Keep a reference to prevent garbage collection
    logo_label.pack(pady=10)

    frame = tk.Frame(signup_window, bg="#FFDAB9")  # Frame for centering widgets
    frame.pack(expand=True)

    new_username_label = tk.Label(frame, text="Username:", font=("Arial", 12), bg="#FFDAB9")
    new_username_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
    new_username_entry = tk.Entry(frame, font=("Arial", 12))
    new_username_entry.grid(row=0, column=1, padx=5, pady=5)

    new_password_label = tk.Label(frame, text="Password:", font=("Arial", 12), bg="#FFDAB9")
    new_password_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
    new_password_entry = tk.Entry(frame, show="*", font=("Arial", 12))
    new_password_entry.grid(row=1, column=1, padx=5, pady=5)

    reenter_password_label = tk.Label(frame, text="Re-enter Password:", font=("Arial", 12), bg="#FFDAB9")
    reenter_password_label.grid(row=2, column=0, padx=5, pady=5, sticky="e")
    reenter_password_entry = tk.Entry(frame, show="*", font=("Arial", 12))
    reenter_password_entry.grid(row=2, column=1, padx=5, pady=5)

    signup_button = tk.Button(frame, text="Sign Up", font=("Arial", 12),
                              command=lambda: signup(new_username_entry, new_password_entry))
    signup_button.grid(row=3, column=0, columnspan=2, padx=5, pady=10)
    login_button = tk.Button(frame, text="Login", font=("Arial", 12), command=open_login_window)
    login_button.grid(row=5, column=0, columnspan=2, padx=5, pady=5)

    # Center the frame
    frame.grid_columnconfigure(0, weight=1)
    frame.grid_columnconfigure(2, weight=1)
    frame.grid_rowconfigure(4, weight=1)


root = tk.Tk()
root.title("StrategiXAnalytics")

# Configure the main window
root.geometry("800x600")
root.configure(background="#FFDAB9")  # Coral Peach color

# Title
title_label = tk.Label(root, text="Document Matching and Keyword Analysis System", font=("Arial", 24), bg="#FFDAB9", fg="#fff")
title_label.pack(pady=20)

# Taskbar
taskbar_frame = tk.Frame(root, bg="#333")
taskbar_frame.pack(fill="x")

# Function to create larger font
larger_font = ("Arial", 14)

home_button = tk.Button(taskbar_frame, text="Home", bg="#333", fg="#fff", bd=0, font=larger_font)
home_button.pack(side="left", padx=50, pady=20)  # Increased padx

graph_button = tk.Button(taskbar_frame, text="Graph", bg="#333", fg="#fff", bd=0, font=larger_font,
                         command=calculate_matching_percentage)
graph_button.pack(side="left", padx=50, pady=20)  # Increased padx

dbot_button = tk.Button(taskbar_frame, text="Dbot", bg="#333", fg="#fff", bd=0, font=larger_font)
dbot_button.pack(side="left", padx=50, pady=20)  # Increased padx

login_button = tk.Button(taskbar_frame, text="Login/Signup", bg="#333", fg="#fff", bd=0, font=larger_font,
                         command=open_signup_window)
login_button.pack(side="right", padx=50, pady=20)  # Increased padx

# Image container
image_frame = tk.Frame(root, bg="#FFDAB9")
image_frame.pack(expand=True)


# Function to center the image
def center_image(event):
    image_label.place(relx=0.5, rely=0.5, anchor="center")


# Load and display the image
pil_image = Image.open("image doc.png")
pil_image = pil_image.resize((400, 300))  # Resize the image if needed
image = ImageTk.PhotoImage(pil_image)
image_label = tk.Label(image_frame, image=image, bg="#FFDAB9")
image_label.bind("<Configure>", center_image)
image_label.pack()

# Container
container_frame = tk.Frame(root, bg="#FFDAB9", width=800, height=100)
container_frame.pack(expand=True)

# Left upload button
left_upload_frame = tk.Frame(container_frame, bg="#FFDAB9")
left_upload_frame.pack(side="left", padx=20)

left_label = tk.Label(left_upload_frame, text="Upload First Document:", bg="#FFDAB9", font=larger_font)
left_label.pack()

left_entry = tk.Entry(left_upload_frame, width=30)
left_entry.pack(pady=5)

left_button = tk.Button(left_upload_frame, text="Browse", command=lambda: browse_file(left_entry), font=larger_font)
left_button.pack(pady=5)

# Right upload button
right_upload_frame = tk.Frame(container_frame, bg="#FFDAB9")
right_upload_frame.pack(side="right", padx=20)

right_label = tk.Label(right_upload_frame, text="Upload Second Document:", bg="#FFDAB9", font=larger_font)
right_label.pack()

right_entry = tk.Entry(right_upload_frame, width=30)
right_entry.pack(pady=5)

right_button = tk.Button(right_upload_frame, text="Browse", command=lambda: browse_file(right_entry), font=larger_font)
right_button.pack(pady=5)

result_label = tk.Label(root, text="", font=("Arial", 12), bg="#FFDAB9")
result_label.pack()
# Matching percentage button
matching_button = tk.Button(root, text="Calculate X Correlation", bg="#333", fg="#fff", bd=0,
                            command=calculate_matching_percentage, font=larger_font)
matching_button.pack(pady=25)


# Create the button for saving highly repeated keywords

def open_chatbot_window():
    # Create the popup window
    chatbot_window = tk.Toplevel(root)
    chatbot_window.title("Dbot Assistant")
    chatbot_window.configure(bg="#FFDAB9")

    # Get the position and size of the chatbot image button
    chatbot_x, chatbot_y = chatbot_label.winfo_rootx(), chatbot_label.winfo_rooty()
    chatbot_width, chatbot_height = chatbot_label.winfo_width(), chatbot_label.winfo_height()

    # Get the size of the chatbot window
    window_width, window_height = 1000, 500  # Adjust these values as needed

    # Calculate the position of the chatbot window
    window_x = chatbot_x + chatbot_width // 2 - window_width // 2 - 200
    window_y = chatbot_y - window_height

    # Position the chatbot window just above the chatbot image button
    chatbot_window.geometry(f"{window_width}x{window_height}+{window_x}+{window_y}")

    # Function to handle user input and generate responses

    def get_response():

        chat_history = ""
        # Get user input
        user_input = input_entry.get("1.0", tk.END).strip()

        left_file_paths = left_entry.get().split('\n')
        right_file_paths = right_entry.get().split('\n')

        # Concatenate document contents with user input for better context
        for left_file in left_file_paths:
            if left_file:
                left_text = read_pdf(left_file)
                chat_history += f"First Document Content:\n{left_text}\n"

        for right_file in right_file_paths:
            if right_file:
                right_text = read_pdf(right_file)
                chat_history += f"Second Document Content:\n{right_text}\n"

        chat_history += f"You: {user_input}\n"
        # Call OpenAI API to generate response
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",  # You can adjust the model according to your preference
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": chat_history}

            ],
            max_tokens=1000  # Adjust max_tokens as needed
        )

        # Extract response from the API result
        chatbot_response = response.choices[0].message.content
        chat_history += f"Chatbot: {chatbot_response}\n"

        # Display response
        response_textbox.config(state=tk.NORMAL)
        response_textbox.delete("1.0", tk.END)
        response_textbox.insert(tk.END, chatbot_response)
        response_textbox.config(state=tk.DISABLED)

    # Create and pack widgets
    response_label = tk.Label(chatbot_window, text=" Dbot:", font=("Times New Roman", 15), bg="#FFDAB9")
    response_label.pack(pady=10)

    response_frame = ttk.Frame(chatbot_window, style='My.TFrame', padding=(5, 5, 5, 5))
    response_frame.pack()

    response_textbox = tk.Text(chatbot_window, height=13, width=100, font=("Arial", 12), state=tk.DISABLED,
                               bg="#FFFFFF")
    response_textbox.pack()

    input_label = tk.Label(chatbot_window, text="Enter your query:", font=("Arial", 15), bg="#FFDAB9")
    input_label.pack(pady=(0, 5))

    input_frame = ttk.Frame(chatbot_window, style='My.TFrame', padding=(8, 8, 8, 8))
    input_frame.pack(pady=10)

    input_frame = tk.Frame(chatbot_window, bg="#FFDAB9")
    input_frame.pack(pady=5)

    input_entry = tk.Text(input_frame, height=2, width=26, font=("Arial", 12), bg="#FFFFFF")
    input_entry.pack(side=tk.LEFT)

    send_button = tk.Button(input_frame, text="⚡", font=("Times New Roman", 15), command=get_response, bg="#F5DEB3",
                            bd=0, highlightthickness=0)
    send_button.pack(padx=(10, 0), pady=(0, 10), side=tk.RIGHT)
    # Start the main event loop
    chatbot_window.mainloop()


# Chatbot image
chatbot_image = Image.open("dbot image.png")
chatbot_image = chatbot_image.resize((50, 50))  # Resize the image if needed
chatbot_image = ImageTk.PhotoImage(chatbot_image)
chatbot_label = tk.Label(root, image=chatbot_image, bg="#FFDAB9")
chatbot_label.place(relx=1, rely=1, anchor="se", x=-40, y=-60)

# Bind the chatbot popup function to the image label
chatbot_label.bind("<Button-1>", lambda event: open_chatbot_window())

root.mainloop()