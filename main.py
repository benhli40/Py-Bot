from transformers import AutoTokenizer, AutoModelForCausalLM
import sqlite3
from typing import Dict, List, Tuple
import json
import re
from datetime import datetime

class EnhancedPythonMentor:
    def __init__(self):
        """
        Initialize the enhanced Python mentor with Q&A capabilities
        """
        # Initialize model for generating responses
        model_name = "Salesforce/codegen-350M-mono"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Initialize the knowledge base
        self.setup_knowledge_base()
        
        # Common programming concepts and their explanations
        self.concept_patterns = {
            r'\bvenv\b|virtual environment': 'virtual_environment',
            r'\bdjango\b': 'django_framework',
            r'\bflask\b': 'flask_framework',
            r'\bapi\b': 'api_development',
            r'\bdeployment\b': 'deployment',
            r'\bdocker\b': 'docker',
            r'\btest(ing)?\b': 'testing',
            r'\bdebug(ging)?\b': 'debugging'
        }

    def setup_knowledge_base(self):
        """
        Set up SQLite database for storing Q&A pairs and user interactions
        """
        self.conn = sqlite3.connect('python_mentor.db')
        self.cursor = self.conn.cursor()
        
        # Create tables for storing Q&A data
        self.cursor.executescript('''
            CREATE TABLE IF NOT EXISTS qa_pairs (
                id INTEGER PRIMARY KEY,
                question TEXT,
                answer TEXT,
                category TEXT,
                difficulty TEXT,
                timestamp DATETIME
            );
            
            CREATE TABLE IF NOT EXISTS user_interactions (
                id INTEGER PRIMARY KEY,
                question TEXT,
                matched_response TEXT,
                helpful BOOLEAN,
                timestamp DATETIME
            );
            
            CREATE TABLE IF NOT EXISTS code_examples (
                id INTEGER PRIMARY KEY,
                concept TEXT,
                code TEXT,
                explanation TEXT,
                difficulty TEXT
            );
        ''')
        
        # Initialize with some basic Q&A pairs if empty
        self.cursor.execute('SELECT COUNT(*) FROM qa_pairs')
        if self.cursor.fetchone()[0] == 0:
            self.initialize_basic_qa()

    def initialize_basic_qa(self):
        """
        Initialize the database with common Python project questions and answers
        """
        basic_qa = [
            {
                'question': 'How do I start a new Python project?',
                'answer': '''Here are the steps to start a new Python project:
                1. Create a new directory for your project
                2. Set up a virtual environment
                3. Initialize git repository
                4. Create requirements.txt
                5. Set up basic project structure''',
                'category': 'project_setup',
                'difficulty': 'beginner'
            },
            {
                'question': 'How do I deploy my Python application?',
                'answer': '''Deployment steps typically include:
                1. Prepare your application (debug=False, environment variables)
                2. Choose a hosting platform (Heroku, AWS, DigitalOcean)
                3. Set up deployment configuration
                4. Deploy your application
                5. Monitor for any issues''',
                'category': 'deployment',
                'difficulty': 'intermediate'
            }
        ]
        
        for qa in basic_qa:
            self.cursor.execute('''
                INSERT INTO qa_pairs (question, answer, category, difficulty, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (qa['question'], qa['answer'], qa['category'], 
                  qa['difficulty'], datetime.now()))
        
        self.conn.commit()

    def analyze_question(self, question: str) -> Dict[str, str]:
        """
        Analyze the question to determine its category and difficulty
        """
        # Convert question to lowercase for better matching
        question_lower = question.lower()
        
        # Identify category based on keywords
        category = 'general'
        for pattern, cat in self.concept_patterns.items():
            if re.search(pattern, question_lower):
                category = cat
                break
        
        # Estimate difficulty based on question complexity
        difficulty = 'beginner'
        if any(word in question_lower for word in ['advanced', 'complex', 'architecture']):
            difficulty = 'advanced'
        elif any(word in question_lower for word in ['deploy', 'optimize', 'secure']):
            difficulty = 'intermediate'
            
        return {'category': category, 'difficulty': difficulty}

    def get_relevant_code_example(self, concept: str) -> str:
        """
        Retrieve a relevant code example for the concept
        """
        self.cursor.execute('''
            SELECT code, explanation 
            FROM code_examples 
            WHERE concept = ? 
            LIMIT 1
        ''', (concept,))
        
        result = self.cursor.fetchone()
        if result:
            return f"```python\n{result[0]}\n```\n{result[1]}"
        return None

    def generate_response(self, question: str) -> Dict:
        """
        Generate a comprehensive response to a question
        """
        # Analyze the question
        analysis = self.analyze_question(question)
        
        # Look for existing similar questions
        self.cursor.execute('''
            SELECT answer 
            FROM qa_pairs 
            WHERE category = ? 
            AND difficulty = ?
            AND question LIKE ?
        ''', (analysis['category'], analysis['difficulty'], f'%{question}%'))
        
        existing_answer = self.cursor.fetchone()
        
        if existing_answer:
            response = existing_answer[0]
        else:
            # Generate new response using the model
            context = f"Python project question: {question}\nDetailed answer:"
            inputs = self.tokenizer.encode(context, return_tensors="pt")
            outputs = self.model.generate(
                inputs,
                max_length=300,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Store the new Q&A pair
            self.cursor.execute('''
                INSERT INTO qa_pairs (question, answer, category, difficulty, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (question, response, analysis['category'], 
                analysis['difficulty'], datetime.now()))
            self.conn.commit()
        
        # Get relevant code example if available
        code_example = self.get_relevant_code_example(analysis['category'])
        
        return {
            'answer': response,
            'code_example': code_example,
            'category': analysis['category'],
            'difficulty': analysis['difficulty'],
            'follow_up_questions': self.generate_follow_up_questions(analysis['category'])
        }

    def generate_follow_up_questions(self, category: str) -> List[str]:
        """
        Generate relevant follow-up questions based on the category
        """
        follow_ups = {
            'virtual_environment': [
                "How do I activate the virtual environment?",
                "Should I include venv in git?"
            ],
            'deployment': [
                "How do I set up environment variables?",
                "What are the best practices for production deployment?"
            ],
            'testing': [
                "How do I write unit tests?",
                "What testing framework should I use?"
            ]
        }
        return follow_ups.get(category, [])

    def record_interaction(self, question: str, response: str, helpful: bool):
        """
        Record user interaction for improving responses
        """
        self.cursor.execute('''
            INSERT INTO user_interactions (question, matched_response, helpful, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (question, response, helpful, datetime.now()))
        self.conn.commit()

# Example usage
if __name__ == "__main__":
    mentor = EnhancedPythonMentor()
    
    # Example interaction
    question = "How do I structure a Flask project for deployment?"
    response = mentor.generate_response(question)
    
    print("\nQuestion:", question)
    print("\nResponse:", json.dumps(response, indent=2))
    
    # Record if the response was helpful
    mentor.record_interaction(question, response['answer'], helpful=True)