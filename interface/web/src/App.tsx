import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import { AppProvider } from '@/hooks/useAppContext';
import { HomeShell } from '@/layouts/HomeShell';
import { ModeShell } from '@/layouts/ModeShell';
import { HomePage } from '@/pages/HomePage';
import { DailyFieldPage } from '@/pages/daily/DailyFieldPage';
import { DailyRunPage } from '@/pages/daily/DailyRunPage';
import { DemoDashboardPage } from '@/pages/demo/DemoDashboardPage';
import { SeasonDemoRunPage } from '@/pages/demo/SeasonDemoRunPage';
import { RecommendationsPage } from '@/pages/RecommendationsPage';
import { TradeoffsPage } from '@/pages/TradeoffsPage';
import { ReportsPage } from '@/pages/ReportsPage';

export default function App() {
  return (
    <AppProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<HomeShell />}>
            <Route index element={<HomePage />} />
          </Route>

          <Route path="/demo" element={<ModeShell mode="demo" />}>
            <Route index element={<Navigate to="/demo/run" replace />} />
            <Route path="run" element={<SeasonDemoRunPage />} />
            <Route path="dashboard" element={<DemoDashboardPage />} />
            <Route path="log" element={<RecommendationsPage />} />
            <Route path="reports" element={<ReportsPage />} />
            <Route path="tradeoffs" element={<TradeoffsPage />} />
          </Route>

          <Route path="/daily" element={<ModeShell mode="daily" />}>
            <Route index element={<Navigate to="/daily/run" replace />} />
            <Route path="run" element={<DailyRunPage />} />
            <Route path="field" element={<DailyFieldPage />} />
          </Route>

          <Route path="simulation" element={<Navigate to="/" replace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AppProvider>
  );
}
