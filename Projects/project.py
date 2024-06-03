class AnonymousSurvey:
    """Collect anonymous answers to a survey question."""

    def __init__(self, question):
        """Store a question, and prepare to store responses."""
        self.question = question
        self.responses = []

    def show_question(self, n):
        """Show the survey question."""
        print(self.question[n])

    def store_response(self, new_response):
        """Store a single response to the survey."""
        self.responses.append(new_response)

    def show_results(self):
        """Show all the responses that have been given."""
        print("Survey results:")
        for response in self.responses:
            print(f"- {response}")


# Define a question, and make a survey.

question1 = "What language did you first learn to speak?"
question2 = "what religion do you have?"
question3 = "What do you want to become?"
question = [question1, question2, question3]
language_survey = AnonymousSurvey(question)

# Show the question, and store responses to the question.

print("Enter 'q' at any time to quit.\n")
# while True:
for n in range(len(question)):
    language_survey.show_question(n)
    response = input("Enter your answer to the questions : ")
    if response == "q":
        break
    language_survey.store_response(response)

# Show the survey results.
print("\nThank you to everyone who participated in the survey!")
language_survey.show_results()
