import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AppLayout } from './components/layout/AppLayout';
import { Overview } from './pages/Overview';
import { Enrolment } from './pages/Enrolment';
import { Authenticate } from './pages/Authenticate';
import { Speakers } from './pages/Speakers';
import { GraphExplorer } from './pages/GraphExplorer';
import { AttackLab } from './pages/AttackLab';
import { Evaluation } from './pages/Evaluation';
import { Corpus } from './pages/Corpus';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      staleTime: 60000,
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<AppLayout />}>
            <Route index element={<Overview />} />
            <Route path="enrolment" element={<Enrolment />} />
            <Route path="authenticate" element={<Authenticate />} />
            <Route path="speakers" element={<Speakers />} />
            <Route path="graph-explorer" element={<GraphExplorer />} />
            <Route path="attack-lab" element={<AttackLab />} />
            <Route path="evaluation" element={<Evaluation />} />
            <Route path="corpus" element={<Corpus />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
