import { readFile } from "fs";

export interface UserRecord {
  name: string;
}

export class UserService {
  load(id: string): UserRecord { return { name: id }; }
}

export function makeUserService(): UserService {
  return new UserService();
}

const makeId = (seed: number): string => `${seed}`;

export { makeId };
