import { expect, it } from "vitest";
import { withoutPictureMarkers } from "./pictureText";

it("removes only the picture-text markers", () => {
  const text = "Parts.\n\n<!-- Start of picture text -->\nEncoder Decoder<br><!-- End of picture text -->\n";
  expect(withoutPictureMarkers(text)).toBe("Parts.\n\n\nEncoder Decoder<br>\n");
  expect(withoutPictureMarkers("Plain <!-- other comment --> text.")).toBe("Plain <!-- other comment --> text.");
});
