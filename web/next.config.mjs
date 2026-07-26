/** @type {import('next').NextConfig} */
// output: "standalone" bundles a minimal server + only the needed node_modules
// into .next/standalone, so the Docker runtime stage stays small.
export default { reactStrictMode: true, output: "standalone" };
