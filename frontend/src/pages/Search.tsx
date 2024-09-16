import { useState } from "react";
import axios from "axios";

interface SearchResult {
	page_content: string;
	source: string;
	similarity: number;
}

export default function SemanticSearch() {
	const [query, setQuery] = useState("");
	const [baseResults, setBaseResults] = useState<SearchResult[]>([]);
	const [fineTunedResults, setFineTunedResults] = useState<SearchResult[]>(
		[],
	);
	const [error, setError] = useState<string | null>(null);

	const handleSubmit = async () => {
		try {
			setBaseResults([]);
			setFineTunedResults([]);
			setError(null);

			const baseResponse = await axios.post(
				import.meta.env.VITE_API_URL + "/search",
				{ query },
			);
			setBaseResults(baseResponse.data.responses.slice(0, 3));

			const fineTunedResponse = await axios.post(
				import.meta.env.VITE_API_URL + "/search-fine-tuned",
				{ query },
			);
			setFineTunedResults(fineTunedResponse.data.responses.slice(0, 3));
		} catch (err) {
			console.error(err);
			setError("Backend not loaded or request failed.");
		}
	};

	return (
		<div className="flex flex-col items-center justify-center p-4">
			<div className="w-full max-w-4xl">
				<h1 className="text-2xl font-bold text-blue-800 mb-4 text-center">
					Semantic Search
				</h1>
				<div className="mb-4">
					<input
						type="text"
						value={query}
						onChange={(e) => setQuery(e.target.value)}
						placeholder="Ask a question..."
						className="w-full p-3 border border-gray-300 rounded-lg mb-4"
					/>
					<button
						onClick={handleSubmit}
						className="w-full bg-blue-800 hover:bg-blue-600 text-white py-2 px-4 rounded-lg"
					>
						Submit
					</button>
				</div>

				{error && (
					<div className="text-red-600 font-semibold text-center mb-4">
						{error}
					</div>
				)}

				<div className="grid grid-cols-2 gap-4">
					<div className="bg-white p-4 rounded-lg shadow">
						<h2 className="text-lg font-semibold text-blue-800">
							Base Model
						</h2>
						{baseResults.length > 0 ? (
							baseResults.map((result, index) => (
								<div key={index} className="mb-4">
									<b>Result {index + 1}</b>
									<p className="text-gray-700">
										<b>Answer:</b> {result.page_content}
									</p>
									<p className="text-gray-700 mt-2">
										<b>Source:</b> {result.source}
									</p>
									<p className="text-gray-700 mt-2">
										<b>Similarity Score:</b>{" "}
										{result.similarity}
									</p>
								</div>
							))
						) : (
							<p className="text-gray-400">No answer yet.</p>
						)}
					</div>

					<div className="bg-white p-4 rounded-lg shadow">
						<h2 className="text-lg font-semibold text-blue-800">
							Fine-tuned Model
						</h2>
						{fineTunedResults.length > 0 ? (
							fineTunedResults.map((result, index) => (
								<div key={index} className="mb-4">
									<b>Result {index + 1}</b>
									<p className="text-gray-700">
										<b>Answer:</b> {result.page_content}
									</p>
									<p className="text-gray-700 mt-2">
										<b>Source:</b> {result.source}
									</p>
									<p className="text-gray-700 mt-2">
										<b>Similarity Score:</b>{" "}
										{result.similarity}
									</p>
								</div>
							))
						) : (
							<p className="text-gray-400">No answer yet.</p>
						)}
					</div>
				</div>
			</div>
		</div>
	);
}
