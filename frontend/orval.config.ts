import { defineConfig } from "orval";

export default defineConfig({
  "backend-api": {
    input: {
      target: "/opt/backend_api_openapi.yml",
    },
    output: {
      target: "./src/orval/backend-api.ts",
      schemas: "./src/orval/models/backend-api",
      client: "swr",
      override: {
        mutator: {
          path: "./src/libs/mutator.ts",
          name: "useMutator",
        },
      },
    },
    hooks: {
      afterAllFilesWrite: ["prettier --write ./src/orval"],
    },
  },
});
