import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { ScoreCard } from './ScoreCard';

describe('ScoreCard', () => {
  it('muestra el score y el chip de calentamiento', () => {
    render(<ScoreCard score={0.123} isFitted={false} isAnomaly={false} />);
    expect(screen.getByText('0.123')).toBeInTheDocument();
    expect(screen.getByText('Calentando')).toBeInTheDocument();
  });

  it('muestra ANOMALÍA cuando corresponde', () => {
    render(<ScoreCard score={0.9} isFitted isAnomaly />);
    expect(screen.getByText('ANOMALÍA')).toBeInTheDocument();
  });
});
