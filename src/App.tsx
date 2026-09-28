import { Routes, Route } from "react-router";
import Layout from "@/components/Layout";
import Home from "@/pages/Home";
import Playground from "@/pages/Playground";
import TutorialList from "@/pages/tutorials/TutorialList";
import TutorialDetail from "@/pages/tutorials/TutorialDetail";

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/playground" element={<Playground />} />
        <Route path="/tutorials" element={<TutorialList />} />
        <Route path="/tutorials/:id" element={<TutorialDetail />} />
      </Routes>
    </Layout>
  );
}
