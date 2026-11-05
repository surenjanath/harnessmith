// CUSTOMIZE: swap these two for whatever heading/body fonts the target
// app's design system uses. Module names must match a folder under
// node_modules/@remotion/google-fonts/ (run `ls node_modules/@remotion/google-fonts`
// to browse, or check https://fonts.google.com for the exact family name).
import { loadFont as loadHeadingFont } from "@remotion/google-fonts/Inter";
import { loadFont as loadBodyFont } from "@remotion/google-fonts/Inter";

const heading = loadHeadingFont();
const body = loadBodyFont();

export const OUTFIT = heading.fontFamily;
export const MANROPE = body.fontFamily;
