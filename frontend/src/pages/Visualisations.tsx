import { useEffect, useState } from "react";
import axios from "axios";

export default function Visualisations() {
	const [baseImages, setBaseImages] = useState<string[]>([]);
	const [fineTunedImages, setFineTunedImages] = useState<string[]>([]);
	const [error, setError] = useState<string | null>(null);
	const [enlargedImage, setEnlargedImage] = useState<string | null>(null);

	useEffect(() => {
		const fetchBasePlots = async () => {
			try {
				const baseResponse = await axios.get(
					`${import.meta.env.VITE_API_URL}/base-plots`,
				);
				setBaseImages(baseResponse.data.images);
			} catch (err) {
				console.error(err);
				setError("Failed to load base model plots.");
			}
		};

		const fetchFineTunedPlots = async () => {
			try {
				const fineTunedResponse = await axios.get(
					`${import.meta.env.VITE_API_URL}/fine-tuned-plots`,
				);
				setFineTunedImages(fineTunedResponse.data.images);
			} catch (err) {
				console.error(err);
				setError("Failed to load fine-tuned model plots.");
			}
		};

		fetchBasePlots();
		fetchFineTunedPlots();
	}, []);

	const handleImageClick = (imageUrl: string) => {
		setEnlargedImage(imageUrl);
	};

	const handleCloseModal = () => {
		setEnlargedImage(null);
	};

	return (
		<div className="p-4">
			<h1 className="text-2xl font-bold text-blue-800 mb-4 text-center">
				Model Visualisations
			</h1>

			{error && (
				<div className="text-red-600 font-semibold text-center mb-4">
					{error}
				</div>
			)}

			<div className="flex justify-between mb-8 text-center">
				<div className="w-1/2 pr-2">
					<h2 className="text-xl font-bold text-blue-600 mb-4">
						Base Model Plots
					</h2>
					{baseImages && baseImages.length > 0 ? (
						baseImages.map((image, index) => (
							<img
								key={index}
								src={`${
									import.meta.env.VITE_API_URL
								}/get-image/${image}`}
								alt={`Base Model Plot ${index + 1}`}
								className="w-full h-auto rounded-lg shadow mb-4 cursor-pointer"
								onClick={() =>
									handleImageClick(
										`${
											import.meta.env.VITE_API_URL
										}/get-image/${image}`,
									)
								}
							/>
						))
					) : (
						<p className="text-gray-500">
							No base model plots found.
						</p>
					)}
				</div>

				<div className="w-1/2 pl-2">
					<h2 className="text-xl font-bold text-blue-600 mb-4">
						Fine-Tuned Model Plots
					</h2>
					{fineTunedImages && fineTunedImages.length > 0 ? (
						fineTunedImages.map((image, index) => (
							<img
								key={index}
								src={`${
									import.meta.env.VITE_API_URL
								}/get-image/${image}`}
								alt={`Fine-Tuned Model Plot ${index + 1}`}
								className="w-full h-auto rounded-lg shadow mb-4 cursor-pointer"
								onClick={() =>
									handleImageClick(
										`${
											import.meta.env.VITE_API_URL
										}/get-image/${image}`,
									)
								}
							/>
						))
					) : (
						<p className="text-gray-500">
							No fine-tuned model plots found.
						</p>
					)}
				</div>
			</div>

			{enlargedImage && (
				<div
					className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 cursor-pointer"
					onClick={handleCloseModal}
				>
					<div className="relative">
						<img
							src={enlargedImage}
							alt="Enlarged plot"
							className="max-w-full max-h-full rounded-lg"
						/>
					</div>
				</div>
			)}
		</div>
	);
}
