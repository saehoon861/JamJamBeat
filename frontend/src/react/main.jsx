import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";

const rootElement = document.getElementById("root");

if (!rootElement) {
  throw new Error("React root element #root was not found.");
}

createRoot(rootElement).render(<App />);
