import type { Config } from 'tailwindcss';

// Tokens reutilizados desde src/dashboard/styles.py::PALETTE para continuidad visual.
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: '#7C3AED',
        anomaly: '#EF4444',
        normal: '#10B981',
        warning: '#F59E0B',
        surface: '#FFFFFF',
        bg: '#F1F5F9',
        ink: '#0F172A',
        muted: '#64748B',
        line: '#E2E8F0',
      },
      fontFamily: {
        sans: ['Inter', 'Segoe UI', 'system-ui', 'sans-serif'],
      },
      keyframes: {
        pulseDot: {
          '0%,100%': { opacity: '1', transform: 'scale(1)' },
          '50%': { opacity: '0.4', transform: 'scale(0.85)' },
        },
      },
      animation: {
        pulseDot: 'pulseDot 1.6s ease-in-out infinite',
      },
    },
  },
  plugins: [],
} satisfies Config;
