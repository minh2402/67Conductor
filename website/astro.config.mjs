// @ts-check
import { defineConfig } from 'astro/config';

import tailwind from '@astrojs/tailwind';

// https://astro.build/config
export default defineConfig({
  integrations: [tailwind()],
  site: 'https://scary-spirit-r5qrx9w455x2p7pj-4321.app.github.dev/',
  // base: 'my-repo',
});