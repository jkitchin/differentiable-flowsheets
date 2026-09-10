/**
 * The published page: the frozen half of the same front end.
 *
 * `difflow.publish` inlines this bundle and drops the model next to it
 * as `window.DIFFLOW`. Nothing here fetches, and nothing here solves.
 */
import { mount } from 'svelte'

import Frozen from './lib/Frozen.svelte'
import './publish.css'

export default mount(Frozen, {
  target: document.getElementById('app'),
  props: { data: window.DIFFLOW },
})
