export { WORKDAY_PROFILE } from './workday.ts';
export { TALEO_PROFILE } from './taleo.ts';
export { ICIMS_PROFILE } from './icims.ts';
export { GREENHOUSE_PROFILE } from './greenhouse.ts';
export { LEVER_PROFILE } from './lever.ts';
export { SUCCESSFACTORS_PROFILE } from './successfactors.ts';
export type { ATSProfile, ATSQuirk } from './types.ts';

import type { ATSProfile } from './types.ts';
import { WORKDAY_PROFILE } from './workday.ts';
import { TALEO_PROFILE } from './taleo.ts';
import { ICIMS_PROFILE } from './icims.ts';
import { GREENHOUSE_PROFILE } from './greenhouse.ts';
import { LEVER_PROFILE } from './lever.ts';
import { SUCCESSFACTORS_PROFILE } from './successfactors.ts';

// all ATS profiles ordered by market share/strictness
export const ALL_PROFILES: ATSProfile[] = [
	WORKDAY_PROFILE,
	TALEO_PROFILE,
	SUCCESSFACTORS_PROFILE,
	ICIMS_PROFILE,
	GREENHOUSE_PROFILE,
	LEVER_PROFILE
];

// lookup a profile by name (case-insensitive)
export function getProfile(name: string): ATSProfile | undefined {
	return ALL_PROFILES.find((p) => p.name.toLowerCase() === name.toLowerCase());
}
