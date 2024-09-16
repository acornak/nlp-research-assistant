import { FC, ReactNode } from "react";

interface LayoutWrapperProps {
	children: ReactNode;
}

export const LayoutWrapper: FC<LayoutWrapperProps> = ({ children }) => {
	return (
		<div className="min-h-screen bg-gray-100">
			<div className="container mx-auto p-6">
				<div className="bg-white shadow-lg rounded-lg p-6">
					{children}
				</div>
			</div>
		</div>
	);
};
