/** Buffer circular de capacidad fija: push O(1), memoria acotada. */
export class RingBuffer<T> {
  private readonly buf: T[] = [];
  constructor(public readonly capacity: number) {}

  push(value: T): void {
    this.buf.push(value);
    if (this.buf.length > this.capacity) this.buf.shift();
  }

  get length(): number {
    return this.buf.length;
  }

  /** Copia inmutable del contenido en orden de inserción. */
  toArray(): T[] {
    return this.buf.slice();
  }

  last(): T | undefined {
    return this.buf[this.buf.length - 1];
  }

  clear(): void {
    this.buf.length = 0;
  }
}
