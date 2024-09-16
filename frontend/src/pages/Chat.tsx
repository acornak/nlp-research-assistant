import { useState } from "react";
import axios from "axios";

export default function Chat() {
	const [query, setQuery] = useState("");
	const [baseAnswer, setBaseAnswer] = useState<string | null>(null);
	const [baseSource, setBaseSource] = useState<string | null>(null);
	const [baseSimilarity, setBaseSimilarity] = useState<string | null>(null);
	const [fineTunedAnswer, setFineTunedAnswer] = useState<string | null>(null);
	const [fineTunedSource, setFineTunedSource] = useState<string | null>(null);
	const [fineTunedSimilarity, setFineTunedSimilarity] = useState<
		string | null
	>(null);
	const [error, setError] = useState<string | null>(null);

	const handleSubmit = async () => {
		try {
			setBaseAnswer(null);
			setFineTunedAnswer(null);
			setError(null);

			const baseResponse = await axios.post(
				import.meta.env.VITE_API_URL + "/chat",
				{ query },
			);
			setBaseAnswer(baseResponse.data.answer);
			setBaseSource(baseResponse.data.source);
			setBaseSimilarity(baseResponse.data.similarity);

			const fineTunedResponse = await axios.post(
				import.meta.env.VITE_API_URL + "/chat-fine-tuned",
				{ query },
			);
			setFineTunedAnswer(fineTunedResponse.data.answer);
			setFineTunedSource(fineTunedResponse.data.source);
			setFineTunedSimilarity(fineTunedResponse.data.similarity);
		} catch (err) {
			console.error(err);
			setError("Backend not loaded or request failed.");
		}
	};

	return (
		<div className="flex flex-col items-center justify-center p-4">
			<div className="w-full max-w-4xl">
				<h1 className="text-2xl font-bold text-blue-800 mb-4 text-center">
					Research Assistant Chat
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
						{baseAnswer ? (
							<>
								<p className="text-gray-700 mt-2">
									<b>Answer:</b> {baseAnswer}
								</p>
								<p className="text-gray-700 mt-4">
									<b>Source:</b> {baseSource}
								</p>
								<p className="text-gray-700 mt-4">
									<b>Similarity Score:</b> {baseSimilarity}
								</p>
							</>
						) : (
							<p className="text-gray-400">No answer yet.</p>
						)}
					</div>

					<div className="bg-white p-4 rounded-lg shadow">
						<h2 className="text-lg font-semibold text-blue-800">
							Fine-tuned Model
						</h2>
						{fineTunedAnswer ? (
							<>
								<p className="text-gray-700 mt-2">
									<b>Answer:</b> {fineTunedAnswer}
								</p>
								<p className="text-gray-700 mt-4">
									<b>Source:</b> {fineTunedSource}
								</p>
								<p className="text-gray-700 mt-4">
									<b>Similarity Score:</b>{" "}
									{fineTunedSimilarity}
								</p>
							</>
						) : (
							<p className="text-gray-400">No answer yet.</p>
						)}
					</div>
				</div>
			</div>
		</div>
	);
}
