import { cleanup } from '@testing-library/react'
import '@testing-library/jest-dom/vitest'
import { afterEach } from 'vitest'

afterEach(() => {
  cleanup()
  // Saved tabs and layout must not leak from one test into the next.
  localStorage.clear()
})
