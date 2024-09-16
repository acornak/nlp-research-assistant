import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import Visualisations from "./pages/Visualisations";
import { Navbar } from "./navbar/Navbar";
import { LayoutWrapper } from "./components/Layout";
import Chat from "./pages/Chat";
import SemanticSearch from "./pages/Search";

function App() {
	return (
		<BrowserRouter>
			<Navbar />
			<LayoutWrapper>
				<Routes>
					<Route path="/" element={<Home />} />
					<Route path="/chat" element={<Chat />} />
					<Route path="/search" element={<SemanticSearch />} />
					<Route
						path="/visualisations"
						element={<Visualisations />}
					/>
				</Routes>
			</LayoutWrapper>
		</BrowserRouter>
	);
}

export default App;
