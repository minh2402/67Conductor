# Astro Starter Kit: Basics


```text
# Gallery lightbox
/src/lib/fetchHero.ts (fetch google sheet and parse) 
    /compoents/HeroGallery.astro (gallery lightbox)
    /pages/index.astro 

# Posts blog (need rerun if new stuff added)
/src/lib/utils.ts (decode ANSI to unicode
	/fetchpost.ts (fetch google sheet and parse) 
    /pages/product/index.astro (all post)
		          /[page]      (pagination create page from parsed)
	        	  /tag   /[tag]   (pagination create tags from parsed)

# Photo Card
/src/compoents/PhotoCard.astro (gallery lightbox)
        /pages/meow.astro   (Gallery)

# Card
/src/compoents/Card.astro (gallery lightbox)
        /pages/index.astro  (Peek on what i did)
```

Astro looks for `.astro` or `.md` files in the `src/pages/` directory. Each page is exposed as a route based on its file name.

There's nothing special about `src/components/`, but that's where we like to put any Astro/React/Vue/Svelte/Preact components.

Any static assets, like images, can be placed in the `public/` directory.
