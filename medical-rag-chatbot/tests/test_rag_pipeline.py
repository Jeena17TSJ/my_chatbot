import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.rag_pipeline import MedicalRAGChatbot


class MedicalRAGChatbotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chatbot = MedicalRAGChatbot.__new__(MedicalRAGChatbot)
        self.chatbot.vectorstore = Mock()
        self.chatbot.llm = Mock()

    @patch("src.rag_pipeline.rerank")
    def test_answer_includes_sources_and_recent_history(self, mock_rerank: Mock) -> None:
        documents = [
            Document(
                page_content="Diabetes is a condition involving blood sugar.",
                metadata={"source": "data/medical.txt"},
            ),
            Document(
                page_content="Common symptoms include excessive thirst.",
                metadata={"source": "data/medical.txt"},
            ),
        ]
        self.chatbot.vectorstore.similarity_search.return_value = documents
        mock_rerank.return_value = documents
        self.chatbot.llm.invoke.return_value = AIMessage(
            content="Diabetes affects blood sugar levels. [1]"
        )

        result = self.chatbot.ask_question(
            "What is diabetes?",
            [{"role": "user", "content": "Tell me about blood sugar."}],
        )

        self.chatbot.vectorstore.similarity_search.assert_called_once_with(
            "What is diabetes?", k=6
        )
        prompt = self.chatbot.llm.invoke.call_args.args[0]
        self.assertIn("Tell me about blood sugar.", prompt)
        self.assertIn("[1] Diabetes is a condition involving blood sugar.", prompt)
        self.assertEqual(
            result.sources,
            [{"id": "1", "name": "medical.txt"}],
        )
        self.assertEqual(result.answer, "Diabetes affects blood sugar levels. [1]")

    @patch("src.rag_pipeline.rerank", return_value=[])
    def test_no_ranked_passages_returns_safe_no_context_answer(self, mock_rerank: Mock) -> None:
        self.chatbot.vectorstore.similarity_search.return_value = []

        result = self.chatbot.ask_question("What is not in the documents?")

        self.assertEqual(result.sources, [])
        self.assertIn("couldn't find relevant information", result.answer)
        self.chatbot.llm.invoke.assert_not_called()
        mock_rerank.assert_called_once_with([], "What is not in the documents?")

    def test_blank_question_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.chatbot.ask_question("  ")

        self.chatbot.vectorstore.similarity_search.assert_not_called()


if __name__ == "__main__":
    unittest.main()
