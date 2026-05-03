import widgetFactory from "widgets";

export class Widget {
  render() { return widgetFactory(); }
}

export function makeWidget() {
  return new Widget();
}

const makeName = (raw) => raw.trim();

export { makeName };
