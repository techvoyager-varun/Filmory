import type { Movie } from "@/types/movie";

/**
 * Curated extras (official poster artwork + editorial synopsis + runtime) for a
 * handful of well-known MovieLens ids. Everything else comes from the catalog
 * file itself.
 */
export const CURATED: Record<number, { runtime: number; poster: string; description: string }> = {
  109487: { runtime: 169, poster: "https://upload.wikimedia.org/wikipedia/en/b/bc/Interstellar_film_poster.jpg", description: "With Earth's crops failing, a former pilot leads a crew through a wormhole near Saturn in search of a new home for humanity — while time itself becomes the cruelest obstacle." },
  79132: { runtime: 148, poster: "https://upload.wikimedia.org/wikipedia/en/2/2e/Inception_%282010%29_theatrical_poster.jpg", description: "A thief who steals corporate secrets through dream-sharing technology is handed the inverse job: plant an idea deep inside a rival heir's subconscious." },
  103249: { runtime: 91, poster: "https://upload.wikimedia.org/wikipedia/en/f/f6/Gravity_Poster.jpg", description: "A medical engineer on her first shuttle mission is stranded in orbit after debris destroys her craft, with only her breath and nerve between her and the void." },
  58559: { runtime: 152, poster: "https://upload.wikimedia.org/wikipedia/en/1/1c/The_Dark_Knight_%282008_film%29.jpg", description: "Gotham's fragile peace collapses when an anarchist in greasepaint proves that the city's hero and its worst instincts are two sides of one coin." },
  296: { runtime: 154, poster: "https://upload.wikimedia.org/wikipedia/en/3/3b/Pulp_Fiction_%281994%29_poster.jpg", description: "Hitmen, a boxer, a gangster's wife and a pair of diner robbers collide across a looping, wickedly funny Los Angeles crime tapestry." },
  318: { runtime: 142, poster: "https://upload.wikimedia.org/wikipedia/en/8/81/ShawshankRedemptionMoviePoster.jpg", description: "A wrongly convicted banker builds an unlikely friendship — and a decades-long plan — inside the walls of Shawshank State Penitentiary." },
  2571: { runtime: 136, poster: "https://upload.wikimedia.org/wikipedia/en/d/db/The_Matrix.png", description: "A restless programmer learns his world is a simulation and joins a ragged crew fighting the machines that farm humanity for power." },
  4993: { runtime: 178, poster: "https://upload.wikimedia.org/wikipedia/en/f/fb/Lord_Rings_Fellowship_Ring.jpg", description: "A reluctant hobbit and eight companions set out across Middle-earth to destroy a ring that corrupts everyone who carries it." },
  1210: { runtime: 131, poster: "https://upload.wikimedia.org/wikipedia/en/b/b2/ReturnOfTheJediPoster1983.jpg", description: "The Rebellion mounts a final strike on the Empire while a young Jedi gambles everything on the belief that his father can still be saved." },
  356: { runtime: 142, poster: "https://upload.wikimedia.org/wikipedia/en/6/67/Forrest_Gump_poster.jpg", description: "An earnest man with an extraordinary knack for being present at history's turning points keeps running back to the woman he loves." },
  4306: { runtime: 90, poster: "https://upload.wikimedia.org/wikipedia/en/7/7b/Shrek_%282001_animated_feature_film%29.jpg", description: "A grumpy ogre and a motormouth donkey strike a bargain with a tiny tyrant to win back a swamp — and rescue a princess with a secret." },
  6377: { runtime: 100, poster: "https://upload.wikimedia.org/wikipedia/en/2/29/Finding_Nemo.jpg", description: "An anxious clownfish crosses an entire ocean, helped by a fish who forgets everything, to find the son taken from the reef." },
  68954: { runtime: 96, poster: "https://upload.wikimedia.org/wikipedia/en/0/05/Up_%282009_film%29.jpg", description: "A widowed balloon salesman finally flies his house to South America — with an over-eager scout accidentally along for the ride." },
  89745: { runtime: 143, poster: "https://upload.wikimedia.org/wikipedia/en/8/8a/The_Avengers_%282012_film%29_poster.jpg", description: "Earth's mismatched heroes are forced into one room, and then one line, when an exiled prince opens a portal above Manhattan." },
  106696: { runtime: 102, poster: "https://upload.wikimedia.org/wikipedia/en/0/05/Frozen_%282013_film%29_poster.jpg", description: "A fearless princess treks into an eternal winter to find the sister whose powers froze their kingdom — and her own heart." },
  111: { runtime: 114, poster: "https://upload.wikimedia.org/wikipedia/en/3/33/Taxi_Driver_%281976_film_poster%29.jpg", description: "A sleepless veteran drifts through the city's night shift until his loneliness curdles into a violent sense of purpose." },
  858: { runtime: 175, poster: "https://upload.wikimedia.org/wikipedia/en/1/1c/Godfather_ver1.jpg", description: "The reluctant youngest son of a crime dynasty is pulled into the family business, and slowly becomes the coldest of them all." },
  1213: { runtime: 146, poster: "https://upload.wikimedia.org/wikipedia/en/7/7b/Goodfellas.jpg", description: "Three decades inside the mob, told at breakneck speed by a wiseguy who loved the life right up until it turned on him." },
  593: { runtime: 118, poster: "https://upload.wikimedia.org/wikipedia/en/8/86/The_Silence_of_the_Lambs_poster.jpg", description: "A trainee agent bargains with an imprisoned cannibal for the insight she needs to catch a killer still at large." },
  1219: { runtime: 109, poster: "https://upload.wikimedia.org/wikipedia/commons/7/76/Psycho_%281960%29_theatrical_poster_%28retouched%29.jpg", description: "A secretary on the run stops for the night at a quiet roadside motel run by a young man and his mother." },
  92259: { runtime: 112, poster: "https://upload.wikimedia.org/wikipedia/en/9/93/The_Intouchables.jpg", description: "A wealthy quadriplegic hires a young man from the projects as his caretaker, and neither of their lives survives the arrangement intact." },
  112552: { runtime: 107, poster: "https://upload.wikimedia.org/wikipedia/en/0/01/Whiplash_poster.jpg", description: "An ambitious drummer and a conductor who believes cruelty makes greatness push each other toward brilliance or ruin." },
  68157: { runtime: 153, poster: "https://upload.wikimedia.org/wikipedia/en/c/c3/Inglourious_Basterds_poster.jpg", description: "A cinema owner's revenge and a squad of saboteurs converge on one very flammable night in occupied Paris." },
  91529: { runtime: 164, poster: "https://upload.wikimedia.org/wikipedia/en/8/83/Dark_knight_rises_poster.jpg", description: "Eight years in hiding end when a masked mercenary cuts Gotham off from the world and hands the city back to itself." },
  7153: { runtime: 201, poster: "https://upload.wikimedia.org/wikipedia/en/4/40/Rotkboxart2.jpg", description: "Armies gather before the black gate while two hobbits crawl the last miles toward a mountain of fire." },
  5952: { runtime: 179, poster: "https://upload.wikimedia.org/wikipedia/en/a/a1/Lord_Rings_Two_Towers.jpg", description: "The broken fellowship fights on three fronts as a wizard's army breaks against the walls of Helm's Deep." },
  60069: { runtime: 98, poster: "https://upload.wikimedia.org/wikipedia/en/4/4c/WALL-E_poster.jpg", description: "The last trash compactor on an abandoned Earth falls in love, and accidentally reminds humanity what it left behind." },
  8961: { runtime: 115, poster: "https://upload.wikimedia.org/wikipedia/en/2/27/The_Incredibles_%282004_animated_feature_film%29.jpg", description: "A family of retired supers is dragged out of suburbia by a fan whose admiration turned inside out." },
  99114: { runtime: 165, poster: "https://upload.wikimedia.org/wikipedia/en/8/8b/Django_Unchained_Poster.jpg", description: "A freed man and a bounty hunter ride into the deepest South to buy back a wife, one bullet at a time." },
  4226: { runtime: 113, poster: "https://upload.wikimedia.org/wikipedia/en/c/c7/Memento_poster.jpg", description: "A man who cannot form new memories hunts his wife's killer using tattoos, polaroids and a truth he keeps rewriting." },
  27773: { runtime: 120, poster: "https://upload.wikimedia.org/wikipedia/en/6/67/Oldboykoreanposter.jpg", description: "Released after fifteen years of unexplained captivity, a man is given five days to learn who took them and why." },
  2959: { runtime: 139, poster: "https://upload.wikimedia.org/wikipedia/en/f/fc/Fight_Club_poster.jpg", description: "An insomniac office worker and a soap salesman start a club that quickly stops being about fighting at all." },
};


export const cdn = (url: string, w: number) =>
  `https://images.weserv.nl/?url=${encodeURIComponent(url.replace(/^https?:\/\//, ""))}&w=${w}&output=webp&q=95&we`;

export type { Movie };
