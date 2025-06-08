import { defineConfig } from "orval";

export default defineConfig({
  "backend-api": {
    input: {
      target: "../backend/api/controllers/backend_api/openapi/openapi.yml",
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
