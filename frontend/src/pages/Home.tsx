import { Link } from "react-router-dom";

export default function Home() {
	return (
		<div className="min-h-60 flex items-center justify-center">
			<div className="p-8 text-justify">
				<h1 className="text-4xl font-bold text-blue-800 mb-4 text-center">
					Welcome to the Research Assistant
				</h1>
				<p className="text-gray-700 text-lg mb-6">
					This project aims to help researchers navigate through
					scientific papers using semantic search, document retrieval,
					and AI-powered question answering. The system is built with
					the latest technologies such as ChromaDB, Hugging Face
					Sentence Transformers (fine-tuned), RAG framework, MLFlow,
					FastAPI, and some nice React + Vite frontend.
				</p>
				<p className="text-gray-700 text-lg mb-6">
					Explore the ability to search through your dataset using
					fine-tuned models and visualizations, or ask questions
					directly to the system to get AI-generated answers based on
					the content of your data.
				</p>
				<p className="text-gray-700 text-lg mb-6">
					You can view the complete source code and contribute to the
					project on GitHub.
				</p>
				<div className="text-center">
					<a
						href="https://github.com/acornak/nlp-research-assistant"
						className="text-blue-500 hover:underline font-semibold"
						target="_blank"
						rel="noopener noreferrer"
					>
						View on GitHub
					</a>
				</div>
				<div className="flex justify-between mt-6">
					<Link to="/chat">
						<button className="text-gray-600 hover:bg-blue-600 font-medium rounded-full bg-blue-800 text-white px-6 py-2">
							Continue to chat
						</button>
					</Link>
					<Link to="/visualisations">
						<button className="text-gray-600 hover:bg-blue-600 font-medium rounded-full bg-blue-800 text-white px-6 py-2">
							Compare models
						</button>
					</Link>
				</div>
			</div>
		</div>
	);
}
