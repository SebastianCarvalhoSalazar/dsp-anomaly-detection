import { describe, expect, it } from 'vitest';

import { RingBuffer } from './ringBuffer';

describe('RingBuffer', () => {
  it('respeta la capacidad y sobrescribe los más antiguos', () => {
    const rb = new RingBuffer<number>(3);
    rb.push(1);
    rb.push(2);
    rb.push(3);
    rb.push(4);
    expect(rb.toArray()).toEqual([2, 3, 4]);
    expect(rb.length).toBe(3);
    expect(rb.last()).toBe(4);
  });

  it('clear vacía el buffer', () => {
    const rb = new RingBuffer<number>(2);
    rb.push(1);
    rb.clear();
    expect(rb.length).toBe(0);
    expect(rb.last()).toBeUndefined();
  });
});
