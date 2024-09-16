import { Link } from "react-router-dom";

export const Navbar = () => {
	return (
		<nav className="bg-white shadow-md">
			<div className="container mx-auto flex justify-between items-center p-4">
				<Link to="/">
					<h1 className="text-2xl font-bold text-blue-900">
						Research Assistant
					</h1>
				</Link>
				<div className="space-x-8 flex">
					<Link
						to="/search"
						className="text-gray-600 hover:text-blue-500 font-medium"
					>
						Semantic Search
					</Link>
					<Link
						to="/chat"
						className="text-gray-600 hover:text-blue-500 font-medium"
					>
						Chat
					</Link>
					<Link
						to="/visualisations"
						className="text-gray-600 hover:text-blue-500 font-medium"
					>
						Visualisations
					</Link>
					<a
						href={import.meta.env.VITE_MLFLOW_URL}
						target="_blank"
						rel="noopener noreferrer"
						className="text-gray-600 hover:text-blue-500 font-medium"
					>
						MLFlow
					</a>
				</div>
			</div>
		</nav>
	);
};
