import streamlit as st
import os
import tempfile
from typing import List, Dict, Any
import json
import re
from datetime import datetime

# Core libraries
import PyPDF2
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# GROQ import with error handling
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    st.error("GROQ library not found. Please install it with: pip install groq")
    GROQ_AVAILABLE = False

# Streamlit page config
st.set_page_config(
    page_title="RAG MCQ Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PDFProcessor:
    """Handles PDF text extraction and preprocessing"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks for better context preservation"""
        if not text.strip():
            return []
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
            
            if i + chunk_size >= len(words):
                break
                
        return chunks

class VectorStore:
    """Manages document embeddings and similarity search"""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.chunks = []
        self.embeddings = None
    
    def add_documents(self, chunks: List[str]):
        """Add document chunks to vector store"""
        if not chunks:
            return
        
        self.chunks = chunks
        
        # Generate embeddings
        with st.spinner("Generating embeddings..."):
            self.embeddings = self.model.encode(chunks)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
    
    def similarity_search(self, query: str, k: int = 5) -> List[str]:
        """Retrieve most similar chunks for a query with optimized size"""
        if self.index is None or not self.chunks:
            return []
        
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Get relevant chunks and optimize total size
        relevant_chunks = [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
        
        # Limit total context size to prevent token overflow
        total_context = ""
        optimized_chunks = []
        max_total_chars = 8000  # Conservative limit for context
        
        for chunk in relevant_chunks:
            if len(total_context + chunk) <= max_total_chars:
                optimized_chunks.append(chunk)
                total_context += chunk + "\n\n"
            else:
                # If this chunk would exceed limit, add a truncated version
                remaining_chars = max_total_chars - len(total_context)
                if remaining_chars > 500:  # Only add if we have decent space left
                    truncated_chunk = chunk[:remaining_chars-100] + "..."
                    optimized_chunks.append(truncated_chunk)
                break
        
        return optimized_chunks

class GroqMCQGenerator:
    """Handles MCQ generation using GROQ LLM with advanced prompt engineering"""
    
    def __init__(self, api_key: str):
        if not GROQ_AVAILABLE:
            raise ImportError("GROQ library is not available")
        
        try:
            self.client = Groq(api_key=api_key)
            self.model = "llama3-70b-8192"  # Using the most capable GROQ model
        except Exception as e:
            st.error(f"Failed to initialize GROQ client: {str(e)}")
            raise
    
    def generate_mcqs(self, context: str, topic: str, num_questions: int, 
                     difficulty: str, question_types: List[str]) -> List[Dict]:
        """Generate MCQs with advanced prompt engineering and batch processing"""
        
        if not context.strip():
            st.error("No context provided for MCQ generation")
            return []
        
        # Check if we need to batch the requests
        batch_size = 3  # Generate 3 questions per batch to stay within token limits
        all_mcqs = []
        
        # Calculate number of batches needed
        num_batches = (num_questions + batch_size - 1) // batch_size
        
        # Progress bar for batch processing
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for batch_idx in range(num_batches):
            # Calculate questions for this batch
            questions_in_batch = min(batch_size, num_questions - len(all_mcqs))
            
            # Update progress
            progress = (batch_idx + 1) / num_batches
            progress_bar.progress(progress)
            status_text.text(f"Generating batch {batch_idx + 1}/{num_batches} ({questions_in_batch} questions)...")
            
            # Optimize context size for this batch
            optimized_context = self._optimize_context_size(context, max_chars=3000)
            
            # Generate MCQs for this batch
            batch_mcqs = self._generate_single_batch(
                optimized_context, topic, questions_in_batch, difficulty, question_types, batch_idx + 1
            )
            
            if batch_mcqs:
                all_mcqs.extend(batch_mcqs)
            else:
                st.warning(f"Failed to generate questions for batch {batch_idx + 1}")
            
            # Small delay between batches to respect rate limits
            if batch_idx < num_batches - 1:  # Don't sleep after the last batch
                import time
                time.sleep(1)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return all_mcqs
    
    def _optimize_context_size(self, context: str, max_chars: int = 3000) -> str:
        """Optimize context size to fit within token limits"""
        if len(context) <= max_chars:
            return context
        
        # Split context into sentences and prioritize the most relevant ones
        sentences = context.split('. ')
        
        # If still too long, take the first portion
        optimized = ""
        for sentence in sentences:
            if len(optimized + sentence + ". ") <= max_chars:
                optimized += sentence + ". "
            else:
                break
        
        return optimized.strip()
    
    def _generate_single_batch(self, context: str, topic: str, num_questions: int, 
                              difficulty: str, question_types: List[str], batch_num: int) -> List[Dict]:
        """Generate a single batch of MCQs"""
        
        # Build optimized prompt for smaller batches
        prompt = self._build_batch_prompt(context, topic, num_questions, difficulty, question_types, batch_num)
        
        try:
            # Make API call with proper error handling
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system", 
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=2000,  # Reduced max tokens for batches
                top_p=0.9,
                stream=False
            )
            
            # Extract response content
            response_content = chat_completion.choices[0].message.content
            
            if not response_content:
                st.error(f"Empty response from GROQ API for batch {batch_num}")
                return []
            
            # Parse MCQs from response
            mcqs = self._parse_mcq_response(response_content)
            return mcqs
            
        except Exception as e:
            # Handle rate limit errors specifically
            if "413" in str(e) or "rate_limit" in str(e).lower():
                st.error(f"Batch {batch_num}: Context still too large. Trying with smaller context...")
                # Try with even smaller context
                smaller_context = self._optimize_context_size(context, max_chars=2000)
                return self._generate_single_batch(smaller_context, topic, num_questions, difficulty, question_types, batch_num)
            else:
                st.error(f"Error generating MCQs for batch {batch_num}: {str(e)}")
                return []
    
    def _get_system_prompt(self) -> str:
        """Comprehensive system prompt for MCQ generation"""
        return """You are an expert educational content creator and assessment specialist with deep expertise in:
- Bloom's Taxonomy and cognitive learning levels
- Question design principles and best practices
- Academic assessment standards
- Subject matter expertise across multiple domains

Your role is to create high-quality, pedagogically sound multiple-choice questions that:
1. Test genuine understanding, not just memorization
2. Have clear, unambiguous question stems
3. Include plausible but incorrect distractors
4. Follow proper MCQ construction guidelines
5. Are appropriate for the specified difficulty level
6. Align with educational standards

You MUST respond with properly formatted JSON only. No additional text or explanations outside the JSON structure."""
    
    def _build_advanced_prompt(self, context: str, topic: str, num_questions: int, 
                              difficulty: str, question_types: List[str]) -> str:
        """Build sophisticated prompt with detailed instructions"""
        
        difficulty_guidelines = {
            "Easy": "Focus on basic recall, recognition, and simple understanding. Questions should test fundamental concepts and direct information from the text.",
            "Medium": "Require application of knowledge, analysis of relationships, and synthesis of information from multiple parts of the context.",
            "Hard": "Demand critical thinking, evaluation, complex analysis, and ability to draw inferences beyond directly stated information."
        }
        
        type_instructions = {
            "Factual": "Test specific facts, dates, names, definitions, and concrete information directly stated in the context.",
            "Conceptual": "Assess understanding of principles, theories, classifications, and relationships between ideas.",
            "Application": "Require applying knowledge to new situations, solving problems, or using information in practical contexts.",
            "Analytical": "Involve breaking down complex information, identifying patterns, comparing/contrasting, or examining cause-and-effect relationships."
        }
        
        selected_type_guidance = "\n".join([f"- {qtype}: {type_instructions[qtype]}" for qtype in question_types])
        
        prompt = f"""
CONTEXT INFORMATION:
{context}

TASK SPECIFICATION:
Generate exactly {num_questions} multiple-choice questions about "{topic}" based strictly on the provided context.

DIFFICULTY LEVEL: {difficulty}
{difficulty_guidelines[difficulty]}

QUESTION TYPES TO INCLUDE:
{selected_type_guidance}

QUALITY REQUIREMENTS:
1. QUESTION STEM: Must be clear, specific, and directly related to the context
2. CORRECT ANSWER: Must be definitively correct based on the provided context
3. DISTRACTORS: Must be plausible but clearly incorrect, avoiding:
   - Obviously wrong answers
   - "All of the above" or "None of the above" options
   - Grammatically inconsistent options
4. CONTEXT GROUNDING: Every question must be answerable from the provided context
5. COGNITIVE ALIGNMENT: Questions must match the specified difficulty level and types

FORMATTING REQUIREMENTS:
Respond with a JSON array where each question follows this exact structure:
{{
  "question": "Complete question stem ending with appropriate punctuation",
  "options": {{
    "A": "First option",
    "B": "Second option", 
    "C": "Third option",
    "D": "Fourth option"
  }},
  "correct_answer": "A|B|C|D",
  "explanation": "Clear explanation of why the correct answer is right and why others are wrong",
  "difficulty": "{difficulty}",
  "type": "Question type from the specified list",
  "context_reference": "Brief reference to the relevant part of the context"
}}

CRITICAL INSTRUCTIONS:
- Use ONLY information from the provided context
- Ensure each question has exactly 4 options (A, B, C, D)
- Make distractors challenging but definitively incorrect
- Vary question stems and avoid repetitive patterns
- Balance questions across specified types if multiple types requested
- Double-check that correct answers are unambiguously correct

Generate the MCQs now:"""
        
    def _build_batch_prompt(self, context: str, topic: str, num_questions: int, 
                           difficulty: str, question_types: List[str], batch_num: int) -> str:
        """Build optimized prompt for batch processing"""
        
        difficulty_guidelines = {
            "Easy": "Focus on basic recall, recognition, and simple understanding.",
            "Medium": "Require application of knowledge and analysis of relationships.",
            "Hard": "Demand critical thinking, evaluation, and complex analysis."
        }
        
        type_instructions = {
            "Factual": "Test specific facts and concrete information.",
            "Conceptual": "Assess understanding of principles and relationships.",
            "Application": "Require applying knowledge to new situations.",
            "Analytical": "Involve breaking down complex information and examining relationships."
        }
        
        selected_type_guidance = "\n".join([f"- {qtype}: {type_instructions[qtype]}" for qtype in question_types])
        
        prompt = f"""
CONTEXT (Batch {batch_num}):
{context}

TASK: Generate exactly {num_questions} multiple-choice questions about "{topic}".

REQUIREMENTS:
- Difficulty: {difficulty} - {difficulty_guidelines[difficulty]}
- Question Types: {selected_type_guidance}
- Base all questions on the provided context
- Each question must have exactly 4 options (A, B, C, D)
- Provide clear explanations for correct answers

FORMAT: Respond with valid JSON array only:
[
  {{
    "question": "Question text?",
    "options": {{
      "A": "Option A",
      "B": "Option B", 
      "C": "Option C",
      "D": "Option D"
    }},
    "correct_answer": "A",
    "explanation": "Why this answer is correct",
    "difficulty": "{difficulty}",
    "type": "{question_types[0] if question_types else 'Factual'}"
  }}
]

Generate {num_questions} questions now:"""
        
        return prompt
    
    def _parse_mcq_response(self, response: str) -> List[Dict]:
        """Parse and validate the MCQ response"""
        try:
            # Clean the response
            response = response.strip()
            
            # Extract JSON if wrapped in code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            # Parse JSON
            mcqs = json.loads(response)
            
            # Validate structure
            validated_mcqs = []
            for i, mcq in enumerate(mcqs):
                if self._validate_mcq_structure(mcq):
                    validated_mcqs.append(mcq)
                else:
                    st.warning(f"Question {i+1} has invalid structure and was skipped")
            
            return validated_mcqs
            
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse MCQ response as JSON: {str(e)}")
            return []
        except Exception as e:
            st.error(f"Error parsing MCQ response: {str(e)}")
            return []
    
    def _validate_mcq_structure(self, mcq: Dict) -> bool:
        """Validate MCQ structure"""
        required_keys = ["question", "options", "correct_answer", "explanation"]
        
        if not all(key in mcq for key in required_keys):
            return False
        
        if not isinstance(mcq["options"], dict):
            return False
        
        if not all(option in mcq["options"] for option in ["A", "B", "C", "D"]):
            return False
        
        if mcq["correct_answer"] not in ["A", "B", "C", "D"]:
            return False
        
        return True

def main():
    st.title("📚 RAG-based MCQ Generator")
    st.markdown("Upload PDFs and generate high-quality multiple-choice questions on any topic!")
    
    # Check GROQ availability
    if not GROQ_AVAILABLE:
        st.error("❌ GROQ library is not installed. Please install it with: `pip install groq`")
        st.stop()
    
    # Initialize session state
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = VectorStore()
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    if 'generated_mcqs' not in st.session_state:
        st.session_state.generated_mcqs = []
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # GROQ API Key
        groq_api_key = st.text_input(
            "GROQ API Key",
            type="password",
            help="Enter your GROQ API key from console.groq.com"
        )
        
        if not groq_api_key:
            st.warning("⚠️ Please enter your GROQ API key to continue")
            st.info("Get your API key from: https://console.groq.com")
            st.stop()
        
        # Test API key
        try:
            test_client = Groq(api_key=groq_api_key)
            st.success("✅ GROQ API key validated")
        except Exception as e:
            st.error(f"❌ Invalid GROQ API key: {str(e)}")
            st.stop()
        
        st.divider()
        
        # MCQ Generation Parameters
        st.subheader("MCQ Parameters")
        
        topic = st.text_input(
            "Topic/Subject",
            placeholder="e.g., Photosynthesis, Machine Learning, World War II",
            help="Specify the topic for MCQ generation"
        )
        
        num_questions = st.slider(
            "Number of Questions",
            min_value=1,
            max_value=20,
            value=5,
            help="How many MCQs to generate"
        )
        
        difficulty = st.selectbox(
            "Difficulty Level",
            ["Easy", "Medium", "Hard"],
            index=1,
            help="Cognitive complexity of questions"
        )
        
        question_types = st.multiselect(
            "Question Types",
            ["Factual", "Conceptual", "Application", "Analytical"],
            default=["Factual", "Conceptual"],
            help="Types of questions to include"
        )
        
        if not question_types:
            st.error("Please select at least one question type")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF files"
        )
        
        if uploaded_files:
            if st.button("Process Documents", use_container_width=True):
                with st.spinner("Processing documents..."):
                    all_chunks = []
                    
                    for uploaded_file in uploaded_files:
                        # Extract text
                        text = PDFProcessor.extract_text_from_pdf(uploaded_file)
                        
                        if text:
                            # Chunk text
                            chunks = PDFProcessor.chunk_text(text)
                            all_chunks.extend(chunks)
                            st.success(f"✅ Processed {uploaded_file.name}")
                        else:
                            st.error(f"❌ Failed to process {uploaded_file.name}")
                    
                    if all_chunks:
                        # Add to vector store
                        st.session_state.vector_store.add_documents(all_chunks)
                        st.session_state.documents_processed = True
                        st.success(f"🎉 Successfully processed {len(all_chunks)} text chunks!")
                    else:
                        st.error("No text could be extracted from the uploaded files")
        
        # Document status
        if st.session_state.documents_processed:
            st.info(f"📊 Ready! {len(st.session_state.vector_store.chunks)} chunks processed")
    
    with col2:
        st.header("🎯 MCQ Generation")
        
        if not st.session_state.documents_processed:
            st.info("Please upload and process documents first")
        elif not topic:
            st.info("Please specify a topic in the sidebar")
        elif not question_types:
            st.info("Please select question types in the sidebar")
        else:
            if st.button("Generate MCQs", use_container_width=True, type="primary"):
                with st.spinner("Generating MCQs..."):
                    try:
                        # Retrieve relevant context
                        relevant_chunks = st.session_state.vector_store.similarity_search(
                            topic, k=10
                        )
                        
                        if not relevant_chunks:
                            st.error("❌ No relevant content found for the specified topic. Try a different topic or upload more relevant documents.")
                        else:
                            # Combine chunks for context
                            context = "\n\n".join(relevant_chunks)
                            context_length = len(context)
                            
                            # Show context info with optimization details
                            st.info(f"📄 Using {len(relevant_chunks)} relevant text chunks ({context_length:,} characters)")
                            
                            # Show batch processing info
                            batch_size = 3
                            num_batches = (num_questions + batch_size - 1) // batch_size
                            
                            if num_questions > batch_size:
                                st.info(f"🔄 Processing in {num_batches} batches of up to {batch_size} questions each to optimize token usage")
                            
                            # Generate MCQs
                            generator = GroqMCQGenerator(groq_api_key)
                            mcqs = generator.generate_mcqs(
                                context, topic, num_questions, difficulty, question_types
                            )
                            
                            if mcqs:
                                st.session_state.generated_mcqs = mcqs
                                st.success(f"✅ Successfully generated {len(mcqs)} MCQs!")
                                
                                # Show batch completion summary
                                if num_questions > batch_size:
                                    st.success(f"🎉 All {num_batches} batches completed successfully!")
                                
                                st.balloons()
                            else:
                                st.error("❌ Failed to generate MCQs. Please check your API key and try again.")
                                st.info("💡 Try reducing the number of questions or selecting a more specific topic.")
                    
                    except Exception as e:
                        st.error(f"❌ Error during MCQ generation: {str(e)}")
                        st.error("Please check your API key and internet connection.")
    
    # Display generated MCQs
    if st.session_state.generated_mcqs:
        st.header("📝 Generated MCQs")
        
        # Export options
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("📄 Export as JSON"):
                json_str = json.dumps(st.session_state.generated_mcqs, indent=2)
                st.download_button(
                    "Download JSON",
                    json_str,
                    f"mcqs_{topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
        
        with col2:
            if st.button("📊 Export as CSV"):
                df_data = []
                for i, mcq in enumerate(st.session_state.generated_mcqs):
                    df_data.append({
                        'Question_No': i + 1,
                        'Question': mcq['question'],
                        'Option_A': mcq['options']['A'],
                        'Option_B': mcq['options']['B'],
                        'Option_C': mcq['options']['C'],
                        'Option_D': mcq['options']['D'],
                        'Correct_Answer': mcq['correct_answer'],
                        'Explanation': mcq['explanation'],
                        'Difficulty': mcq.get('difficulty', ''),
                        'Type': mcq.get('type', ''),
                        'Context_Reference': mcq.get('context_reference', '')
                    })
                
                df = pd.DataFrame(df_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    f"mcqs_{topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
        
        with col3:
            if st.button("🔄 Clear MCQs"):
                st.session_state.generated_mcqs = []
                st.rerun()
        
        st.divider()
        
        # Display MCQs
        for i, mcq in enumerate(st.session_state.generated_mcqs):
            with st.expander(f"Question {i + 1} - {mcq.get('type', 'Unknown')} ({mcq.get('difficulty', 'Unknown')})"):
                st.markdown(f"**{mcq['question']}**")
                
                # Options
                for option, text in mcq['options'].items():
                    if option == mcq['correct_answer']:
                        st.success(f"✅ {option}. {text}")
                    else:
                        st.write(f"{option}. {text}")
                
                # Explanation
                st.info(f"**Explanation:** {mcq['explanation']}")
                
                # Additional metadata
                if mcq.get('context_reference'):
                    st.caption(f"**Context:** {mcq['context_reference']}")

if __name__ == "__main__":
    main()