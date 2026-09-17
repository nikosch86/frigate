/**
 * Review/events page object.
 *
 * Encapsulates severity tab, filter bar, calendar, and mobile filter
 * drawer selectors. Does NOT own assertions.
 */

import type { Locator, Page } from "@playwright/test";
import { BasePage } from "./base.page";

export class ReviewPage extends BasePage {
  constructor(page: Page, isDesktop: boolean) {
    super(page, isDesktop);
  }

  get alertsTab(): Locator {
    return this.page.getByLabel("Alerts");
  }

  get detectionsTab(): Locator {
    return this.page.getByLabel("Detections");
  }

  get motionTab(): Locator {
    return this.page.getByRole("radio", { name: "Motion" });
  }

  get camerasFilterTrigger(): Locator {
    return this.page.getByRole("button", { name: /cameras/i }).first();
  }

  get calendarTrigger(): Locator {
    return this.page.getByRole("button", { name: /24 hours|calendar|date/i });
  }

  get showReviewedToggle(): Locator {
    return this.page.getByRole("button", { name: /reviewed/i });
  }

  get reviewItems(): Locator {
    return this.page.locator(".review-item");
  }

  /** Review-list thumbnails carrying the given camera friendly name. */
  thumbnailsLabelled(cameraName: string): Locator {
    return this.reviewItems.filter({
      has: this.page.getByText(cameraName, { exact: true }),
    });
  }

  /** The recording view header row (Back/Live buttons + title or logo). */
  get recordingHeader(): Locator {
    return this.page.locator("div.h-11").filter({
      has: this.page.getByRole("button", {
        name: "Go to the main camera live view",
      }),
    });
  }

  /** The centered camera friendly-name label in the recording header. */
  get recordingHeaderName(): Locator {
    return this.recordingHeader.locator(".capitalize");
  }

  /** The filter popover content (desktop) or drawer (mobile). */
  get filterOverlay(): Locator {
    return this.page
      .locator(
        '[data-radix-popper-content-wrapper], [role="dialog"], [data-vaul-drawer]',
      )
      .first();
  }
}
